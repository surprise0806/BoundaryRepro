# Blind task attribution

This task is a clean-room, dependency-free reproduction inspired by the public
[`python-dotenv` issue #207](https://github.com/theskumar/python-dotenv/issues/207).
The repository snapshot is original BoundaryRepro fixture code and does not
copy upstream implementation code.

Only `task.json`, `repository/`, its public tests, and public command outputs
may enter agent context. `hidden_tests/` is outside the copied workspace and is
invoked only by the trusted verifier. No upstream patch, fixing commit, or
answer explanation is packaged with this task.
