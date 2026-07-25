# Contributing

1. Create a branch for each change.
2. Keep the V8.3 main release unchanged unless a change has been verified on the actual rover.
3. Put unverified navigation or detection variants under `experiments/`.
4. Run `scripts/syntax_check.sh` before committing.
5. Document hardware-test conditions and observed behavior in the pull request.
6. Never commit model weights, private keys, passwords, access tokens, or large sensor recordings.
