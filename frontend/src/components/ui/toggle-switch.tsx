"use client";

import clsx from "clsx";

export interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  subtitle?: string;
  disabled?: boolean;
  id?: string;
}

export function ToggleSwitch({ checked, onChange, label, subtitle, disabled, id }: ToggleSwitchProps) {
  return (
    <div className={clsx("inline-flex items-center gap-2.5", disabled && "opacity-40")}>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={clsx(
          "relative h-5 w-9 shrink-0 rounded-full border transition-colors duration-fast focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2",
          checked ? "border-ink bg-ink" : "border-line bg-ink/[0.08]",
          disabled ? "cursor-not-allowed" : "cursor-pointer",
        )}
      >
        <span
          className={clsx(
            "absolute top-[2px] h-3.5 w-3.5 rounded-full bg-surface transition-transform duration-fast shadow-xs",
            checked ? "translate-x-[16px] rtl:-translate-x-[16px]" : "translate-x-[2px] rtl:-translate-x-[2px]",
          )}
        />
      </button>
      {(label || subtitle) ? (
        <label htmlFor={id} className={clsx("flex flex-col text-start", disabled ? "cursor-not-allowed" : "cursor-pointer")}>
          {label ? <span className="text-sm font-medium text-ink">{label}</span> : null}
          {subtitle ? <span className="text-xs leading-[16px] text-ink-tertiary">{subtitle}</span> : null}
        </label>
      ) : null}
    </div>
  );
}
