import sys

if len(sys.argv) < 2:
    print(
        f'[-] Only one commnad required which is in ("train", "inference", "interactive", "train_tokenizer)',
        file=sys.stderr,
    )
    exit(-1)

_, command, *arguments = sys.argv

if command == "train":
    from .train import main, parser
elif command == "inference":
    from .inference import main, parser
elif command == "interactive":
    from .interactive import main, parser
elif command == "train_tokenizer":
    from .train_tokenizer import main, parser
else:
    print(f'[-] Please type command in ("train", "inference", "interactive", "train_tokenizer)', file=sys.stderr)
    exit(-1)

exit(main(parser.parse_args(arguments)))
