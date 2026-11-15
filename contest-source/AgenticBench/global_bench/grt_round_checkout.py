"""Host handler for an in-call parent checkout; not an MCP server or budget ledger."""
import copy
import fcntl
from pathlib import Path

from grt_bench.archive import Archive
from grt_bench.common import digest, read, write
from grt_bench.contract import validate
from .grt_round_prompt import check_parent_request


class RoundParentCheckout:
    """Bind one prompt, round and destination; never accept host paths from the model.

    The controller exposes checkout through its restricted tool bridge and owns
    isolation, writable-file permissions and the campaign's evaluation budget.
    A durable pending record prevents replaying uncertain checkout side effects.
    """

    def __init__(self, root, task, bundle, control, workspace):
        self.root = Path(root).resolve()
        self.task, self.bundle = copy.deepcopy(task), copy.deepcopy(bundle)
        self.control, self.workspace = Path(control).resolve(), Path(workspace).resolve()
        self.control.relative_to(self.root / "control")
        self.workspace.relative_to(self.root / "workspaces")
        if self.control == self.root / "control" or self.workspace == self.root / "workspaces":
            raise ValueError("Use dedicated round control and workspace directories")
        if validate(task) != bundle["context"]["task_digest"]:
            raise ValueError("Checkout task differs from prompt")
        self.binding = {"bundle_digest": digest(bundle), "task_digest": validate(task),
                        "workspace": str(self.workspace)}

    def checkout(self, proposal):
        check_parent_request(self.bundle, proposal)
        proposal = copy.deepcopy(proposal)
        self.control.mkdir(parents=True, exist_ok=True)
        with (self.control / "parent.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state_path = self.control / "parent.json"
            if state_path.exists():
                state = read(state_path)
                if state["binding"] != self.binding or state["proposal"] != proposal:
                    raise ValueError("This round is already bound to another checkout request")
                if state["status"] != "ready":
                    raise ValueError("Uncertain parent checkout requires host reconciliation")
                if not self.workspace.is_dir():
                    raise ValueError("Prepared workspace is missing; host reconciliation required")
                return state["receipt"]
            if self.workspace.exists():
                raise ValueError("Parent workspace must be new")
            state = {"binding": self.binding, "proposal": proposal, "status": "pending"}
            write(state_path, state)
            archive = Archive(self.root, self.task)
            selection = archive.select(proposal["parent"], proposal["reason"])
            # Retain selection even if materialization is interrupted.
            state["selection"] = selection
            write(state_path, state, replace=True)
            checkout = archive.checkout(self.workspace, selection["id"])
            receipt = {"bundle_digest": digest(self.bundle), "task_digest": checkout["task_digest"],
                       "round": self.bundle["context"]["budget"]["round_number"],
                       "parent": checkout["parent"], "selection_id": checkout["selection"],
                       "workspace": self.bundle["context"]["workspace"], "proposal": proposal}
            state.update(status="ready", receipt=receipt)
            write(state_path, state, replace=True)
            return receipt

    def receipt(self):
        """Read host evidence after the model exits, never its self-reported copy."""
        path = self.control / "parent.json"
        if not path.exists():
            return None
        state = read(path)
        if state["binding"] != self.binding or state["status"] != "ready":
            raise ValueError("Parent checkout requires host reconciliation")
        return state["receipt"]
