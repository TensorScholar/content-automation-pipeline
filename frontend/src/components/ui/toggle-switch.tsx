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
                        ? "border-brand/30 bg-brand shadow-[inset_0_1px_0_rgb(255_255_255/0.16),0_8px_16px_-14px_rgb(0_0_0/0.55)]"
                        : "border-black/10 bg-black/[0.06] dark:border-white/12 dark:bg-white/[0.08]",
                )}
            >
                <span
                    className={clsx(
                        "absolute top-0.5 h-5.5 w-5.5 rounded-full bg-white shadow-[0_1px_2px_rgb(0_0_0/0.15)] transition-transform duration-150 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)]",
                        checked ? "translate-x-[24px] rtl:-translate-x-[24px]" : "translate-x-0.5 rtl:-translate-x-0.5",
                    )}
                />
            </button>
            {(label || subtitle) && (
                <div className="flex flex-col">
                    {label && <span className="text-[13px] font-medium text-ink">{label}</span>}
                    {subtitle && <span className="text-[12px] text-ink-tertiary">{subtitle}</span>}
                </div>
            )}
        </label>
    );
}
