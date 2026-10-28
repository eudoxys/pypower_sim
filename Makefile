PACKAGE=$(notdir $(PWD))
LOGO="https://github.com/eudoxys/.github/blob/main/eudoxys_banner.png?raw=true"
LINK="https://www.eudoxys.com/"

all: docs test

docs: $(SOURCE)
	test -d .venv || python3 -m venv .venv
	(source .venv/bin/activate ; pip install --upgrade pip)
	(source .venv/bin/activate ; pip install --upgrade pdoc . -r requirements.txt)
	(source .venv/bin/activate ; pdoc $(PACKAGE)/__init__.py -o $@ --logo $(LOGO) --math --mermaid --logo-link $(LINK))

test:
	test -d .venv || python3 -m venv .venv
	(source .venv/bin/activate ; pip install -e . pytest ; pytest)

.PHONY: docs test
