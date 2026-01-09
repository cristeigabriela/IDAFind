# IDAFind

**Pictured:** a simple example of IDAFind, with fuzzy search, highlight found lines, and case sensitive/insensitive search.
![Short, GIF demo of IDAFind for non-collapsed presentation](./media/Demo.gif)
> For a more detailed demo, including shortcut action configuration, please see here:
> <details>
> TODO
> </details>

## What is this?

This is a plugin for IDA that implements rich, responsive search functionality to pseudocode widgets, making searching for what you're looking for less cumbersome.

## Features

- Save your search settings: your last search settings are saved in your **.idb** file, including your highlight color!
- Doesn't block focus: you can keep your search on top at all times, and press your hotkey **(default: Ctrl+F)** to regrab focus on the fly.
- Immediate search: search as you write, and never leave your searchbox. Go back and forth with **Ctrl+Enter**/**Enter** respectively.
- Transparent on unfocus: because we don't block focus, you can switch back to your pseudocode window seamlessly. To make that experience even better, we offer the option to make the window see-through as you do that.
- Wildcard search: functionally the same as the `-Like` functionality of PowerShell. Most Windows programmers are familiar with it already.
- Case insensitive search: does not account for character case in matching.
- **Highlights**: as you type, all the matched lines of your search query (including wildcard search ones) will be highlighted. To help you focus, **the active match (as seen in the search window) pops out, as other matches are dimmed**. You can even pick your preferred color!


## AI disclaimer
I put together most of this with Claude. Well, like 80% of it. It was great at early iteration but at some point it was faster for me to start implementing and writing some of the stuff, and handle refactoring. It was kind of really awesome though.

## License
[The Unlicense/PUBLIC DOMAIN](./LICENSE)