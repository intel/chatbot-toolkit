# @intel-chatbot-toolkit-ui/markdown

This package provides a React component for markdown parsing and rendering in the Intel® Chatbot Toolkit UI. It is based on `marked` and `highlight.js` packages along with components and styles that are a part of `@intel-chatbot-toolkit-ui` pnpm workspace.

## Table of Contents

- [Usage](#usage)
- [License](#license)

## Usage

You can use this package in apps or other packages within the workspace.

Add the package as a dependency in `package.json`:

```json
{
  "dependencies": {
    "@intel-chatbot-toolkit-ui/markdown": "workspace:*"
  }
}
```

Rebuild pnpm workspace to include new dependency:

```bash
pnpm install && pnpm -r build
```

Import `style.scss` from this package into your main SCSS file:

```scss
@import "@intel-chatbot-toolkit-ui/markdown/style.scss";
```

Import `Markdown` component into your code:

```js
import { Markdown } from "@intel-chatbot-toolkit-ui/markdown";

// Usage in a React component
function MyMarkdown() {
  return <Markdown text={"# Hello World"} />;
}
```

## License

See [`LICENSE`](../../../../LICENSE) for license information.
