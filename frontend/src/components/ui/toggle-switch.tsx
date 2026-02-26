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
        <label htmlFor={id} className={clsx("inline-flex cursor-pointer items-center gap-3", disabled && "opacity-50 cursor-not-allowed")}>
            <button
                id={id}
                type="button"
                role="switch"
                aria-checked={checked}
                disabled={disabled}
                onClick={() => !disabled && onChange(!checked)}
                className={clsx(
                    "relative h-6 w-11 rounded-full transition-colors duration-normal",
                    checked ? "bg-brand" : "bg-ink-tertiary",
                )}
            >
                <span
                    className={clsx(
                        "absolute top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform duration-normal",
                        checked ? "translate-x-[22px] rtl:-translate-x-[22px]" : "translate-x-0.5 rtl:-translate-x-0.5",
                    )}
                />
            </button>
            {(label || subtitle) && (
                <div className="flex flex-col">
                    {label && <span className="text-body-md text-ink">{label}</span>}
                    {subtitle && <span className="text-body-sm text-ink-tertiary">{subtitle}</span>}
                </div>
            )}
        </label>
    );
}
