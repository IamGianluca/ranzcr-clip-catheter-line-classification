format:
	isort . && \
	black -l 79 .

submit:
	kaggle competitions submit -c ranzcr-clip-catheter-line-classification -f subs/submission.csv -m "$(COMMENT)"

pep8_checks:
	flake8 src/ pipe/

type_checks:
	mypy src/ pipe/ --ignore-missing-imports
