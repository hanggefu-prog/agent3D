"""One auditable full-rewrite OpenEvolve configuration for local tasks."""
from dataclasses import asdict

MODELS = ("gpt-5.6-sol", "qwen3.8-max", "deepseek-v4-pro")


def configuration(model):
    from openevolve.config import Config, LLMModelConfig
    if model not in MODELS:
        raise ValueError("Model is outside the approved comparison")
    config = Config(diff_based_evolution=False)
    config.max_iterations = 30
    config.random_seed = 42
    db = config.database
    db.population_size = db.archive_size = 5
    db.num_islands = 1
    db.exploitation_ratio, db.exploration_ratio, db.elite_selection_ratio = .7, .2, .1
    db.feature_dimensions = ["complexity", "diversity"]
    db.feature_bins = 10
    db.random_seed = 42
    db.migration_rate = 0
    config.prompt.num_top_programs = 2
    config.prompt.num_diverse_programs = 1
    config.prompt.use_template_stochasticity = False
    config.llm.models = [LLMModelConfig(name=model, weight=1, temperature=.6,
                                       max_tokens=8192, timeout=1800, retries=0)]
    config.llm.evaluator_models = []
    config.evaluator.use_llm_feedback = False
    config.evaluator.max_retries = 0
    config.evaluator.parallel_evaluations = 1
    return config


def validate_configuration(config, model, rounds, *, consumed_generation_slots=0):
    if (type(consumed_generation_slots) is not int or not 0 <= consumed_generation_slots < 30
            or type(rounds) is not int or rounds + consumed_generation_slots != 30
            or asdict(config) != asdict(configuration(model))):
        raise ValueError("Formal comparison requires the exact shared 30-slot configuration")
    return asdict(config)


class RecordedBackend:
    """Durable selection/population evidence around the native ask/tell backend."""
    def __init__(self, backend, root, status_writer):
        self.backend, self.root, self.status_writer = backend, root, status_writer
        self.requests = self.completed = 0

    def ask(self, context):
        from agentic3d.knowledge import write_json
        self.requests += 1
        write_json(self.root / ("ask-%03d.json" % self.requests), {
            "evaluations": context.evaluations, "remaining_budget": context.remaining_budget,
            "population_ids": sorted(self.backend.database.programs) if self.backend.database else [],
        })
        return self.backend.ask(context)

    def tell(self, feedback):
        from agentic3d.knowledge import write_json
        self.backend.tell(feedback)
        self.completed += len(feedback)
        programs = self.backend.database.programs
        if len(programs) > 5:
            raise RuntimeError("Native OpenEvolve exceeded the declared population cap")
        write_json(self.root / ("population-%03d.json" % self.completed), {
            "population_size": len(programs), "maximum_population": 5,
            "programs": {key: value.to_dict() for key, value in programs.items()},
        })
        self.status_writer(feedback)

    def close(self):
        self.backend.close()
