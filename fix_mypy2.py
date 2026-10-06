with open('tests/unit/test_io.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('pa.DataFrameSchema(  # type: ignore[no-untyped-call]{\"val\"', 'pa.DataFrameSchema(  # type: ignore[no-untyped-call]\n        {\"val\"')
text = text.replace('pa.DataFrameSchema(  # type: ignore[no-untyped-call]columns=', 'pa.DataFrameSchema(  # type: ignore[no-untyped-call]\n        columns=')
with open('tests/unit/test_io.py', 'w', encoding='utf-8') as f: f.write(text)
