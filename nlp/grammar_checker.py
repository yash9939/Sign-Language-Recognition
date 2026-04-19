import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def correct(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)