# make documentation

PACKAGE=$(notdir $(PWD))

SOURCE=$(wildcard $(PACKAGE)/*.py)
LOGO="https://github.com/eudoxys/.github/blob/main/eudoxys_banner.png?raw=true"
LINK="https://www.eudoxys.com/"

docs: $(SOURCE)
	echo $(PACKAGE)
	pip install --upgrade pdoc
	pdoc $(SOURCE) -o $@ --logo $(LOGO) --mermaid --logo-link $(LINK)
