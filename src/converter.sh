# First replace unicode characters with plain english (must be done directly in Atom)
sed -i s/ϕ/phi/g Roadways.py
sed -i s/θ/theta/g Roadways.py
sed -i s/Δ/delta/g Roadways.py

# For some reason, most of the commands below don't work with sed. They have
# to be done through the Atom regex find and replace engine
# Remove comments
sed -i s/^\s*\#[^\n]*$//g Roadways.py
sed -i s/\s*\#[^\n]*$//g Roadways.py

# Remove quotes
PATTERN = # "["]+[^"]*["]+"
REPLACEMENT = ""

# Remove whitespace
PATTERN = "^[\s]*[\n]"
REPLACEMENT = ""

# Clean up struct names
PATTERN = "[a-z_ @]*struct \b([\w]*)[\w<:{},# 0-9]*[\n]"
REPLACEMENT = "class $1:\n\tdef __init__(self,):\n"

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

# Rename functions to python methods
PATTERN = "^function\b(.*$)"
REPLACEMENT = "\ndef$1:\n\tpass"

# Remove typing in function definitions (must be run multiple times)
PATTERN = "\b(def[\s]*[\w]*\([\w,]*)::[^,^\)]*\b(.*)"
REPLACEMENT  ="$1$2"

# Move arguments into keyword arguments
PATTERN = "\(self,[\w\s]*\):[\s]*self.\b([\w\d_]*)"
REPLACEMENT = "(self,$1=None):\n\t\tself.$1 = $1"

PATTERN = "\(self,\b([\w\s=]*)\):[\s]*(^.*=.*$)+[\s]*self.\b(.*)"

# Duplicate from "__init__" to the end of the data members
PATTERN = "def \b(__init__\(self,\):[\s\w.]*)\n\n"
REPLACEMENT = "def $1\n$1\n\n"

# Repeatedly concatenate lines
PATTERN = "self\.\b(.*)\s*self\.\b(.*)\n\n"
REPLACEMENT = "self.$1,$2\n\n"

# Throw arguments in (finally!)
PATTERN = "\b(def __init__\(self,)\):\s*\b([\s\w.]+)\b(__init__\(self,\):)[\s]*self\.\b([\w,]*)"
REPLACEMENT = "$1$4):\n\t\t$2"

# Change arguments to keyword arguments (default = None)
PATTERN = ",\b([\w]*)"
REPLACEMENT = ",$1=None"

# Add assignment operator to each data member in constructor
PATTERN = "self\.\b([\w_]*)"
REPLACEMENT = "self.$1 = $1"
