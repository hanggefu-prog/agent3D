"""Lower small read-only lookup tables to the routing scalar language.

Tables are evaluated once where declared. Index bounds are proven before C++
generation, so the hot path needs neither allocation nor unchecked indexing.
"""

import ast
import copy


MAX_TABLE_ITEMS = 64
MAX_TOTAL_TABLE_ITEMS = 256
TEMP_PREFIX = "open3d_table_local_"


def lower_bounded_tables(node, context_fields, static_range):
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id.startswith(TEMP_PREFIX):
            raise ValueError("score_net_order uses a reserved translator name")

    class Lower(ast.NodeTransformer):
        def __init__(self):
            self.tables = {}
            self.loop_ranges = {}
            self.total = 0

        def visit_Name(self, child):
            if child.id in self.tables:
                raise ValueError("score_net_order tables are read-only; use indexing or len")
            return child

        def visit_Call(self, child):
            if isinstance(child.func, ast.Name) and child.func.id == "len":
                if (len(child.args) == 1 and not child.keywords
                        and isinstance(child.args[0], ast.Name)
                        and child.args[0].id in self.tables):
                    return ast.copy_location(
                        ast.Constant(len(self.tables[child.args[0].id][1])), child
                    )
                raise ValueError("score_net_order len requires a declared fixed table")
            return self.generic_visit(child)

        def visit_For(self, child):
            child.iter = self.visit(child.iter)
            start, stop, step, _ = static_range(child.iter)
            if not isinstance(child.target, ast.Name):
                raise ValueError("score_net_order loop target must be one local name")
            name = child.target.id
            if name in self.tables or name in self.loop_ranges:
                raise ValueError("score_net_order cannot reuse table or active loop variable")
            self.loop_ranges[name] = range(start, stop, step)
            child.body = [self.visit(row) for row in child.body]
            child.orelse = [self.visit(row) for row in child.orelse]
            del self.loop_ranges[name]
            return child

        def visit_Subscript(self, child):
            if not isinstance(child.ctx, ast.Load):
                raise ValueError("score_net_order context and tables are read-only")
            index = child.slice
            # Python 3.8 still wraps simple subscription indices.
            if isinstance(index, ast.Index):
                index = index.value
            if isinstance(child.value, ast.Name) and child.value.id == "ctx":
                if isinstance(index, ast.Constant) and index.value in context_fields:
                    return ast.copy_location(
                        ast.Attribute(ast.Name("ctx", ast.Load()), index.value, ast.Load()), child
                    )
                raise ValueError("score_net_order may access only declared ctx fields with literal keys")
            if not isinstance(child.value, ast.Name) or child.value.id not in self.tables:
                raise ValueError("score_net_order indexing requires a declared fixed table or ctx")
            keys, values = self.tables[child.value.id]
            if keys is not None:
                if not isinstance(index, ast.Constant) or index.value not in keys:
                    raise ValueError("score_net_order dictionary index must be a declared literal string key")
                return ast.copy_location(copy.deepcopy(values[keys.index(index.value)]), child)
            if isinstance(index, ast.Name) and index.id in self.loop_ranges:
                indices = list(self.loop_ranges[index.id])
                if any(not -len(values) <= item < len(values) for item in indices):
                    raise ValueError("score_net_order loop index can exceed table bounds")
                if not indices:
                    return ast.copy_location(ast.Constant(0.0), child)
                result = copy.deepcopy(values[indices[-1] % len(values)])
                for item in reversed(indices[:-1]):
                    result = ast.IfExp(
                        ast.Compare(copy.deepcopy(index), [ast.Eq()], [ast.Constant(item)]),
                        copy.deepcopy(values[item % len(values)]), result
                    )
                return ast.copy_location(result, child)
            try:
                value = ast.literal_eval(index)
            except (ValueError, TypeError):
                value = None
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("score_net_order table index must be an integer literal or bounded loop variable")
            if not -len(values) <= value < len(values):
                raise ValueError("score_net_order table index out of bounds")
            return ast.copy_location(copy.deepcopy(values[value % len(values)]), child)

        def table(self, statement):
            name = statement.targets[0].id
            if name in {"ctx", "math", "range", "len", "abs", "min", "max", "float", "round"}:
                raise ValueError("score_net_order cannot assign reserved table name")
            stores = [n for n in ast.walk(node)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name]
            if len(stores) != 1:
                raise ValueError("score_net_order tables cannot be reassigned or used as loop variables")
            literal = statement.value
            keys = None
            if isinstance(literal, ast.Dict):
                if any(not isinstance(k, ast.Constant) or not isinstance(k.value, str)
                       for k in literal.keys):
                    raise ValueError("score_net_order table dictionary keys must be literal strings")
                keys = [k.value for k in literal.keys]
                if len(keys) != len(set(keys)):
                    raise ValueError("score_net_order table dictionary keys must be unique")
                elements = literal.values
            else:
                elements = literal.elts
            if not 1 <= len(elements) <= MAX_TABLE_ITEMS:
                raise ValueError("score_net_order tables require 1 to 64 numeric entries")
            self.total += len(elements)
            if self.total > MAX_TOTAL_TABLE_ITEMS:
                raise ValueError("score_net_order tables exceed 256 total entries")
            rows, values = [], []
            for i, element in enumerate(elements):
                value = self.visit(element)
                target = TEMP_PREFIX + str(self.total) + "_" + str(i)
                rows.append(ast.copy_location(
                    ast.Assign([ast.Name(target, ast.Store())], value), statement
                ))
                values.append(ast.Name(target, ast.Load()))
            self.tables[name] = keys, values
            return rows

    lower = Lower()
    body = []
    for statement in node.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, (ast.Tuple, ast.List, ast.Dict))):
            body.extend(lower.table(statement))
        else:
            body.append(lower.visit(statement))
    node.body = body
    return ast.fix_missing_locations(node)
