# make documentation

PACKAGE=$(notdir $(PWD))

SOURCE=$(wildcard $(PACKAGE)/*.py)
LOGO="https://github.com/eudoxys/.github/blob/main/eudoxys_banner.png?raw=true"
LINK="https://www.eudoxys.com/"

all: docs test

docs: $(SOURCE)
	test -d .venv || python3 -m venv .venv
	(source .venv/bin/activate ; pip install --upgrade pip)
	(source .venv/bin/activate ; pip install --upgrade pdoc . -r requirements.txt)
	(source .venv/bin/activate ; pdoc $(SOURCE) -o $@ --logo $(LOGO) --mermaid --logo-link $(LINK))

test:
	(cd ./test ; source test.sh)

.PHONY: test # force test to rebuild always
