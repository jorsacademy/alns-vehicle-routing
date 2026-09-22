from pyvrp_cvrp.cli import main


def test_cli_reports_solution_summary(capsys):
    main()
    output = capsys.readouterr().out

    assert "PyVRP CVRP demo" in output
    assert "Customers: 6" in output
    assert "Total demand: 27" in output
    assert "Fleet capacity: 30" in output
    assert "Feasible: True" in output
    assert "Objective:" in output
    assert "Distance:" in output
    assert "Routes used:" in output
