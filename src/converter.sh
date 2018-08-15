
#


# Replace struct with type
PATTERN = "[a-z_ @]*struct \b([A-Za-z]*)[A-Za-z<:{},\ 0-9]*"
REPLACEMENT = ""


#
PATTERN = "__init__\(self[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[,]*[.a-zA-Z_,]*\):"
REPLACEMENT = "__init__(self,$1,$2,$3,$4,$5,$6,$7):\n\t\tself.$1\n\t\tself.$2\n\t\tself.$3\n\t\tself.$4\n\t\tself.$5\n\t\tself.$6\n\t\tself.$7"
