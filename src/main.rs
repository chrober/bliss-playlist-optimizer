// SPDX-License-Identifier: GPL-3.0-only

const PROGRAM: &str = "bliss-playlist-optimizer";
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> &'static str {
    "Usage: bliss-playlist-optimizer version [--json]"
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command] if command == "version" => println!("{PROGRAM} {VERSION}"),
        [command, format] if command == "version" && format == "--json" => {
            println!(
                "{{\"schema_version\":1,\"program\":\"{PROGRAM}\",\"version\":\"{VERSION}\",\"core_api\":\"0.1\"}}"
            );
        }
        _ => {
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_mentions_the_supported_command() {
        assert!(usage().contains("version"));
    }
}
