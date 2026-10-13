import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface UISelectOption {
  value: string;
  label: string;
}

interface UISelectProps {
  value: string;
  onChange: (value: string) => void;
  options: UISelectOption[];
  placeholder?: string;
  className?: string;
}

export const UISelect: React.FC<UISelectProps> = ({
  value,
  onChange,
  options,
  placeholder,
  className = "",
}) => {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className={className} aria-label={placeholder}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent position="popper" sideOffset={4}>
        {options.map((opt) => (
          <SelectItem key={opt.value} value={opt.value}>
            {opt.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};
