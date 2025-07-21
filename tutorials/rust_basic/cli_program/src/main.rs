use std::env;
use std::process;

use cli_program::Config;

fn main() {
    let args: Vec<String> = env::args().collect();

    let config: Config = Config::new(&args).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {err}");
        process::exit(1);
    });
    println!(
        "Searching for {} in file {}",
        config.query, config.file_path
    );

    if let Err(e) = cli_program::run(config) {
        eprintln!("Application error: {e}");
        process::exit(1);
    }
}
