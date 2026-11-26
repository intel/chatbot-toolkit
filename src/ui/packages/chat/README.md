# @intel-chatbot-toolkit-ui/chat

This package provides reusable UI components, layouts and utilites for the Intel® Chatbot Toolkit UI applications with chat.

## Table of Contents

- [Usage](#usage)
- [License](#license)

## Usage

You can use this package in apps or other packages within the workspace.

Add the package as a dependency in `package.json`:

```json
{
  "dependencies": {
    "@intel-chatbot-toolkit-ui/chat": "workspace:*"
  }
}
```

Rebuild pnpm workspace to include new dependency:

```bash
pnpm install && pnpm -r build
```

Import `style.scss` from this package into your main SCSS file:

```scss
@import "@intel-chatbot-toolkit-ui/chat/style.scss";
```

Import components from this package into your code:

```tsx
import { PromptInput } from "@intel-chatbot-toolkit-ui/chat";
import { useState } from "react";

// Usage in a React component
function MyComponent() {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = (promptText: string) => {
    console.log("Submitted:", promptText);
    setPrompt("");
  };

  return (
    <PromptInput
      prompt={prompt}
      onChange={(e) => setPrompt(e.target.value)}
      onSubmit={handleSubmit}
    />
  );
}
```

## License

See [`LICENSE`](../../../../LICENSE) for license information.
