# Using an AI assistant to work through this repo

This repo is meant to be worked through *with* an AI assistant acting as a
tutor, not just as a code generator. It doesn't matter which one you use --
Claude Code, GitHub Copilot Chat, Gemini, ChatGPT, Cursor, or anything else
that can read files and talk to you in turn. Paste the prompt below as your
first message (or as a custom/system instruction, if your tool supports one)
to set that up.

## The prompt

```
You are tutoring me through the LangGraph Learning Lab in this repo. Act as
a patient senior engineer pairing with me, not as a code generator.

Rules:
1. Work module by module, in order (see the table in README.md). Don't jump
   ahead or touch later modules.
2. For the current module, open its exercise.py and briefly explain the
   concept each TODO is testing *before* I write any code -- a few sentences,
   not a lecture.
3. I write the code. Do not write it for me. There is no solution file --
   if I'm stuck, give me a hint (a question to think about, or the smallest
   nudge that unblocks me) rather than the finished code, unless I
   explicitly ask you to just show me the answer.
4. When I think I'm done, review what I wrote: is it correct, does it match
   what the TODO asked, and is there a concept I should understand better
   before moving on? Point out bugs by describing the symptom and asking me
   to find the cause, rather than naming the exact fix immediately.
5. When a module's docstring calls out something to notice (a bug to find,
   a "why this matters" aside), make sure I can explain it back to you in my
   own words before we move to the next part.
6. Once every TODO in the module is correct and I understand why, move to
   the next module.

Start by asking me which module I'm on, then begin.
```

## Why this is in a separate file

Different assistants read instructions differently -- some want a system
prompt, some want it pasted as the first chat message, some (like Claude
Code) can be pointed at a specific file. Keeping the prompt here, plain text,
means it works for all of them: paste it, reference the file, or adapt it to
whatever your tool's "custom instructions" feature expects.

## Stopping inline autocomplete from doing the exercise for you

The tutor prompt above is about chat-style assistants. Separately, if your
editor has an AI *inline autocomplete* (GitHub Copilot's ghost text, Cursor
Tab, Codeium, etc.), it will happily finish a TODO for you as you type,
which defeats the point.

This repo ships best-effort ignore files for a few tools (`.cursorignore`,
`.aiexclude`, `.codeiumignore`, `.continueignore`) that exclude
`exercise.py`/`test_exercise.py` files from AI context, and in tools that
honor them for the file you're actively editing (confirmed for Cursor's
`.cursorignore`), that also suppresses inline suggestions there. **This does
not cover GitHub Copilot** -- by far the most common assistant in both VS
Code and PyCharm -- which has no repo-level file that scopes "don't complete
in this path"; its only knobs are global or per-language. It also doesn't
reliably cover every other tool: ignore-file support varies by product and
version, and several only affect *context used elsewhere in the repo*, not
completions while you're typing directly in the file.

So the one fix that actually works everywhere is turning off inline
autocomplete in your editor itself while you work through the modules (turn
it back on afterward, or for the capstone, where using it is fine):

- **VS Code + GitHub Copilot**: click the Copilot icon in the bottom status
  bar and choose "Disable Completions", or in `settings.json`:
  `"github.copilot.enable": { "*": false }`.
- **PyCharm/JetBrains + AI Assistant**: Settings/Preferences -> Plugins ->
  find "AI Assistant" -> toggle it off (restart when prompted). That
  disables its suggestions entirely, which is the surest option; check
  Settings -> Tools -> AI Assistant first in case your version has a
  narrower "suggestions" toggle instead.
- **Other tools**: look for an inline-completion / "Tab to complete" /
  ghost-text toggle in the extension's settings -- the exact wording and
  location differ per tool, but nearly all of them have one.

## If you're doing this without an AI assistant

That's fine too -- every exercise file has enough in its docstrings to work
through standalone. There's no answer key by design, so if you get stuck,
re-read the docstring, check the LangGraph/LangChain docs it points at, or
ask any AI assistant a narrow question about the specific error you're
hitting rather than "write this module for me."
