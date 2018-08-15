
# First replace unicode characters with plain english
PATTERN = "ϕ"
REPLACEMENT = "phi"
PATTERN = "θ"
REPLACEMENT = "theta"

# Remove comments
PATTERN = "^[\s]*[#][^\n]*[\n]"
REPLAEMENT = "\n"
PATTERN = "[\s]*[#][^\n]*[\n]"
REPLAEMENT = "\n"

# Remove quotes
PATTERN = # "["]+[^"]*["]+"
REPLACEMENT = ""

# Remove whitespace
PATTERN = "^[\s]*[\n]"
REPLAEMENT = ""

# Clean up struct names
PATTERN = "[a-z_ @]*struct \b([\w]*)[\w<:{},# 0-9]*[\n]"
REPLACEMENT = "class $1:\n\tdef __init__(self;):\n"

# Remove end
PATTERN = "::\b([\w\s0-9{}]*[^\n^:]*)[\n]end"
REPLACEMENT = "::$1\n"

# Change data members
PATTERN = "[\n][\s]*\b([\w_]+)::\b([\w0-9{},]*)[^\n]*[#]*\b([^\n]*)"
REPLACEMENT = "\n\t\tself.$1"

# Remove everything that's not a class, def, self, or function
PATTERN = "^((?!self)(?!def)(?!class)(?!function).)*$"
REPLACEMENT = ""

# Remove whitespace again
PATTERN = "^[\s]*[\n]"
REPLAEMENT = ""

# Add whitespace between class definitions
PATTERN = "^\b(class.*$)"
REPLACEMENT = "\n$1"

# Rename functions
PATTERN = "^function\b(.*$)"
REPLACEMENT = "\ndef$1:\n\tpass"

#
PATTERN = "__init__\(self[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[\n\s,]*\b([.a-zA-Z_]*)[,]*[.a-zA-Z_,]*\):"
REPLACEMENT = "__init__(self,$1,$2,$3,$4,$5,$6,$7):\n\t\tself.$1\n\t\tself.$2\n\t\tself.$3\n\t\tself.$4\n\t\tself.$5\n\t\tself.$6\n\t\tself.$7"
