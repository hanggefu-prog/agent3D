from scripts import launch_sta_knowledge_matrix as launch


def test_four_cases_have_isolated_runs_and_own_reference():
    for case in launch.TARGETS:
        command, output = launch.command_for(case)
        assert command[command.index("--protocol") + 1] == "knowledge-v2"
        assert command[command.index("--knowledge-phase") + 1] == "independent"
        assert command[command.index("--rounds") + 1] == "5"
        assert command[command.index("--model") + 1] == "gpt-5.6-sol"
        assert output.parts[-2:] == (case, "agentic3d")
        reference = command[command.index("--reference-run") + 1]
        assert case in reference
        assert command[command.index("--remote-comparison") + 1].endswith("/" + case)
        assert "sta-knowledge-v2" in command[command.index("--remote-comparison") + 1]
        assert "--knowledge-import" not in command
        assert "--resume" not in command
    assert launch.command_for("bp", resume=True)[0][-1] == "--resume"


def test_admission_preserves_resource_and_gpu_checks():
    text = (launch.ROOT / "deploy/compute_pool/sta_knowledge_admission.sh").read_text()
    assert "67108864" in text and "20971520" in text and "12288" in text
    assert "flock 7" in text
    assert 'exec bash "$helper" "$@"' in text
