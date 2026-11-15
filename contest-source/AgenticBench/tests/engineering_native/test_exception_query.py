from local_query import EXCEPTIONS, exception_query_digest, reusable_exception
from engineering_bench.protocol import sha


def test_native_constant_exclusion_precedes_target_ranking():
    assert 'sta::find_timing' in EXCEPTIONS
    assert 'sta::pin_sim_logic_value $pin' in EXCEPTIONS
    assert '$value eq "0" || $value eq "1"' in EXCEPTIONS
    assert 'constant-propagation-$value' in EXCEPTIONS
    # No constraints, timing arcs or constant values are changed to create paths.
    assert 'set_case_analysis' not in EXCEPTIONS
    assert 'set_disable_timing' not in EXCEPTIONS


def test_exception_cache_binds_query_implementation(tmp_path):
    file = tmp_path / 'excluded.tsv'
    file.write_text('pin\treason\nb/A\tconstant-propagation-1\n')
    record = {'status': 'succeeded', 'result_sha256': sha(file)}
    assert not reusable_exception(record, file)
    record['exception_query_sha256'] = 'old-implementation'
    assert not reusable_exception(record, file)
    record['exception_query_sha256'] = exception_query_digest()
    assert reusable_exception(record, file)
    file.write_text('pin\treason\n')
    assert not reusable_exception(record, file)
