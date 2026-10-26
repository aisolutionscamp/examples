"""Command line: `python -m password_tool generate` / `python -m password_tool check`."""
import argparse
import getpass
import json
from dataclasses import asdict

from .checker import check_password
from .generator import DEFAULT_ALPHABET, entropy_bits, generate_passphrase, generate_password, load_wordlist
from .hibp import PwnedPasswordsClient


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="password_tool")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="create a random passphrase or password")
    gen.add_argument("--words", type=int, default=6)
    gen.add_argument("--chars", type=int, help="random characters instead of words")
    chk = sub.add_parser("check", help="check a password (read from a hidden prompt)")
    chk.add_argument("--mfa", action="store_true", help="password is one factor of MFA (min. 8)")
    chk.add_argument("--offline", action="store_true", help="skip the Have I Been Pwned lookup")
    chk.add_argument("--user-input", action="append", default=[], help="name, email etc. to penalize")
    args = parser.parse_args(argv)

    if args.command == "generate":
        if args.chars:
            secret, bits = generate_password(args.chars), entropy_bits(args.chars, len(DEFAULT_ALPHABET))
        else:
            secret, bits = generate_passphrase(args.words), entropy_bits(args.words, len(load_wordlist()))
        print(secret)
        print(f"about {bits:.0f} bits of entropy")
        return

    password = getpass.getpass("Password: ")
    result = check_password(
        password,
        min_length=8 if args.mfa else 15,
        user_inputs=args.user_input,
        breach_checker=None if args.offline else PwnedPasswordsClient(),
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
