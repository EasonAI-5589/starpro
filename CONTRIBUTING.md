# Contributing

Thanks for helping improve STAR-Pro. The current public scope is the audited
LLaVA-1.5 and LLaVA-NeXT overlay.

Before opening a pull request:

1. keep changes within the public release scope;
2. run `bash scripts/check_public_release.sh`;
3. run the syntax and entry-point checks in the [evaluation guide](docs/evaluation.md#local-release-checks);
4. document the model, benchmark, nominal token budget, and exact command for
   behavior-changing changes; and
5. do not commit generated answers, submission files, checkpoints, credentials,
   private host paths, or cluster job specifications.

Bug reports should include a minimal command, software versions, expected and
observed behavior, and a complete traceback with private paths and tokens
redacted. Use the security process for anything involving a credential or other
sensitive information.
