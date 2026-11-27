// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { IconButton, Tooltip } from "@intel-chatbot-toolkit-ui/components";
import { PressEvent } from "react-aria-components";

interface LogoutButtonProps {
  onPress: (e: PressEvent) => void;
}

export const LogoutButton = ({ onPress }: LogoutButtonProps) => (
  <Tooltip
    title="Logout"
    trigger={
      <IconButton
        data-testid="logout-button"
        icon="logout"
        aria-label="Logout"
        onPress={onPress}
      />
    }
    placement="bottom"
  />
);
