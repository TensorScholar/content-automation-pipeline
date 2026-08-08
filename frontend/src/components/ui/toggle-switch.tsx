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
                    "relative h-7 w-12 rounded-full border transition-[background-color,border-color,box-shadow] duration-150 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)] focus-visible:outline-none",
                    checked
                        ? "border-brand bg-brand"
                        : "border-black/10 bg-black/[0.075] dark:border-white/10 dark:bg-white/[0.09]",
                )}
            >
                <span
                    className={clsx(
                        "absolute top-0.5 h-5.5 w-5.5 rounded-full bg-white shadow-[0_2px_5px_rgb(0_0_0/0.2)] transition-transform duration-150 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)]",
                        checked ? "translate-x-[24px] rtl:-translate-x-[24px]" : "translate-x-0.5 rtl:-translate-x-0.5",
                    )}
                />
            </button>
            {(label || subtitle) && (
                <div className="flex flex-col">
                    {label && <span className="text-[13px] font-semibold text-ink">{label}</span>}
                    {subtitle && <span className="mt-0.5 text-[11px] leading-4 text-ink-tertiary">{subtitle}</span>}
                </div>
            )}
        </label>
    );
}
