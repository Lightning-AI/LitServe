def test_new_pipe(lit_server):
    pool_size = lit_server.max_pool_size
    for i in range(pool_size):
        pool = lit_server.new_pipe()

    assert len(lit_server.pipe_pool) == 0
    assert len(lit_server.new_pipe()) == 2
