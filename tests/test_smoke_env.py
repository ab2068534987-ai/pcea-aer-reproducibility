from pcos.core.entities import default_cluster_specs
from pcos.core.entities import Task, Workflow
from pcos.env.power_envelope import PowerEnvelopeProvider
from pcos.env.scheduler_env import SchedulerEnv
from pcos.baselines.heuristics import run_policy, heft_policy


def test_heft_smoke():
    cluster = default_cluster_specs()
    env = SchedulerEnv(cluster=cluster, envelope_provider=PowerEnvelopeProvider(sum(s.full_power() for s in cluster)))
    wf = Workflow(
        workflow_id="unit_smoke",
        tasks=[
            Task(
                id=0,
                cpu=2.0,
                gpu=0.0,
                mem=4.0,
                base_duration=10.0,
                profile_name="cpu_preproc",
                deadline=60.0,
            )
        ],
        edges=[],
        metadata={"makespan_target": 60.0},
        makespan_target=60.0,
    )
    metrics = run_policy(env, wf, heft_policy)
    assert metrics['energy_active_total'] >= 0
    assert metrics['makespan'] >= 0
