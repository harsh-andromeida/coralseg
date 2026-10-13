import React from "react";
import { cn } from "@/lib/utils";

interface ThreeDotsLoaderProps {
  size?: "sm" | "md" | "lg";
  color?: string;
  className?: string;
}

const SIZE_CONFIGS = {
  sm: {
    container: "gap-1.5 h-4",
    dot: "w-2 h-2",
  },
  md: {
    container: "gap-2 h-6",
    dot: "w-2.5 h-2.5",
  },
  lg: {
    container: "gap-2.5 h-8",
    dot: "w-3.5 h-3.5",
  },
};

export const ThreeDotsLoader: React.FC<ThreeDotsLoaderProps> = ({
  size = "md",
  color = "#0d7c85",
  className = "",
}) => {
  const config = SIZE_CONFIGS[size] || SIZE_CONFIGS.md;

  return (
    <div
      className={cn(
        "three-dots-loader inline-flex items-center justify-center",
        config.container,
        className
      )}
      role="status"
      aria-label="Loading..."
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className={cn(
            "three-dots-dot rounded-full shrink-0 shadow-2xs",
            config.dot
          )}
          style={{
            backgroundColor: color,
            animation: "threeDotsWave 1.3s infinite ease-in-out both",
            animationDelay: `${(i - 2) * 0.16}s`,
          }}
        />
      ))}
    </div>
  );
};
