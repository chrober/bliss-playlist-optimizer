# Synthetic parity fixture

This fixture contains no real library metadata, filesystem paths, or analysis
results. Regenerate it with `python generate_fixture.py`.

`manifest.json` records artifact hashes and a complete Python-oracle command.
The playlist uses Lyrion's three-line extended-M3U entry form: `#EXTURL`,
`#EXTINF`, and the absolute path.

Generated oracle output is ignored; the source fixture and later reviewed
expected-result snapshots are the versioned parity inputs.

After running the command recorded in `manifest.json`, execute `python verify_python_oracle.py`. The verifier compares only portable, privacy-safe values and allows no repeat-window violations.
