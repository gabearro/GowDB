//! `granular` — a SQL shell over the engine.
//!
//! ```text
//!   granular                          in-memory REPL
//!   granular --data ./db              persistent REPL
//!   granular -q "SELECT 1"            one shot
//!   granular --data ./db -f setup.sql run a script
//!   echo "SELECT 1" | granular        piped input
//! ```

use std::io::{self, BufRead, IsTerminal, Read, Write};

use granular::common::Result;
use granular::Session;

const HELP: &str = "\
granular — hybrid OLAP + OLTP database

USAGE:
    granular [OPTIONS]

OPTIONS:
    --data <DIR>     open a persistent database in DIR (default: in-memory)
    -q, --query SQL  run SQL and exit
    -f, --file PATH  run the statements in PATH and exit
    -h, --help       show this message

REPL COMMANDS:
    .help            this message
    .tables          list tables
    .schema TABLE    show a table's DDL
    .stats TABLE     compression and index footprint
    .quit / .exit    leave (Ctrl-D also works)

Statements are terminated by `;`. A statement may span multiple lines.";

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dir: Option<String> = None;
    let mut oneshot: Option<String> = None;
    let mut file: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(());
            }
            "--data" => {
                i += 1;
                dir = args.get(i).cloned();
            }
            "-q" | "--query" => {
                i += 1;
                oneshot = args.get(i).cloned();
            }
            "-f" | "--file" => {
                i += 1;
                file = args.get(i).cloned();
            }
            other => {
                eprintln!("unknown argument `{other}`\n\n{HELP}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let mut session = match &dir {
        Some(d) => Session::open(d)?,
        None => Session::in_memory(),
    };

    if let Some(path) = file {
        let sql = std::fs::read_to_string(&path)?;
        run_script(&mut session, &sql);
        session.checkpoint()?;
        return Ok(());
    }

    if let Some(sql) = oneshot {
        run_and_print(&mut session, &sql);
        session.checkpoint()?;
        return Ok(());
    }

    // Piped input: treat the whole stream as a script.
    if !io::stdin().is_terminal() {
        let mut sql = String::new();
        io::stdin().read_to_string(&mut sql)?;
        run_script(&mut session, &sql);
        session.checkpoint()?;
        return Ok(());
    }

    repl(&mut session, dir.as_deref())
}

fn run_and_print(session: &mut Session, sql: &str) {
    match session.run(sql) {
        Ok(results) => {
            for r in results {
                println!("{r}");
            }
        }
        Err(e) => eprintln!("error [{}]: {e}", e.code()),
    }
}

/// Feed a script through the same line handling the REPL uses, so `.tables`
/// and friends work identically whether typed, piped, or read from a file.
/// Returns true if a `.quit` was seen.
fn run_script(session: &mut Session, text: &str) -> bool {
    let mut buffer = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if buffer.trim().is_empty() && trimmed.starts_with('.') {
            match dot_command(session, trimmed) {
                Ok(true) => return true,
                Ok(false) => {}
                Err(e) => eprintln!("error [{}]: {e}", e.code()),
            }
            continue;
        }
        buffer.push_str(line);
        buffer.push('\n');
        if !buffer.trim_end().ends_with(';') {
            continue;
        }
        let sql = std::mem::take(&mut buffer);
        if !sql.trim().is_empty() {
            run_and_print(session, &sql);
        }
    }
    // A trailing statement without its semicolon is still worth running.
    if !buffer.trim().is_empty() {
        run_and_print(session, &buffer);
    }
    false
}

fn repl(session: &mut Session, dir: Option<&str>) -> Result<()> {
    println!(
        "granular {} — {} database. `.help` for commands, `.quit` to exit.",
        env!("CARGO_PKG_VERSION"),
        if dir.is_some() { "persistent" } else { "in-memory" }
    );

    let stdin = io::stdin();
    let mut buffer = String::new();

    loop {
        let prompt = if buffer.trim().is_empty() { "granular> " } else { "     ...> " };
        print!("{prompt}");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break; // EOF
        }
        let trimmed = line.trim();

        // Dot commands only apply at the start of a fresh statement.
        if buffer.trim().is_empty() && trimmed.starts_with('.') {
            match dot_command(session, trimmed) {
                Ok(true) => break,
                Ok(false) => {}
                Err(e) => eprintln!("error [{}]: {e}", e.code()),
            }
            continue;
        }

        buffer.push_str(&line);
        // Wait for a terminating `;` so multi-line statements work.
        if !buffer.trim_end().ends_with(';') {
            continue;
        }
        let sql = std::mem::take(&mut buffer);
        if !sql.trim().is_empty() {
            run_and_print(session, &sql);
        }
    }

    session.checkpoint()?;
    Ok(())
}

/// Returns `Ok(true)` when the REPL should exit.
fn dot_command(session: &mut Session, cmd: &str) -> Result<bool> {
    let mut parts = cmd.splitn(2, char::is_whitespace);
    let head = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim();

    match head {
        ".quit" | ".exit" => return Ok(true),
        ".help" => println!("{HELP}"),
        ".tables" => println!("{}", session.query("SHOW TABLES")?),
        ".schema" => {
            if arg.is_empty() {
                eprintln!("usage: .schema TABLE");
            } else {
                println!("{}", session.query(&format!("SHOW CREATE TABLE {arg}"))?);
            }
        }
        ".stats" => {
            if arg.is_empty() {
                eprintln!("usage: .stats TABLE");
            } else {
                use granular::sql::ast::ObjectName;
                session.catalog.flush_all()?;
                let name = ObjectName(arg.split('.').map(|s| s.to_string()).collect());
                let t = session.catalog.table(&name)?;
                println!("{}", t.compression_report());
                println!(
                    "  parts: {}, buffered writes: {}",
                    t.part_count(),
                    t.delta_len()
                );
            }
        }
        other => eprintln!("unknown command `{other}` — try .help"),
    }
    Ok(false)
}
