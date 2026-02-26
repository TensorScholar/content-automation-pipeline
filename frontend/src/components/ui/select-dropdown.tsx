"use client";
import { forwardRef, useId, useRef, useState, useEffect, useCallback } from "react";
import clsx from "clsx";

export interface SelectOption { value: string; label: string; icon?: string; }

export interface SelectDropdownProps {
    label: string;
    options: SelectOption[];
    value?: string;
    onChange?: (value: string) => void;
    helperText?: string;
    errorText?: string;
    disabled?: boolean;
    placeholder?: string;
    required?: boolean;
    id?: string;
}

export const SelectDropdown = forwardRef<HTMLDivElement, SelectDropdownProps>(function SelectDropdown(
    { label, options, value, onChange, helperText, errorText, disabled, placeholder, required, id: providedId },
    ref
) {
    const autoId = useId();
    const id = providedId ?? autoId;
    const [open, setOpen] = useState(false);
    const [focusIndex, setFocusIndex] = useState(-1);
    const containerRef = useRef<HTMLDivElement>(null);
    const listRef = useRef<HTMLUListElement>(null);
    const hasError = Boolean(errorText);
    const selected = options.find(o => o.value === value);

    const close = useCallback(() => { setOpen(false); setFocusIndex(-1); }, []);

    useEffect(() => {
        if (!open) return;
        const handler = (e: MouseEvent) => {
            if (!containerRef.current?.contains(e.target as Node)) close();
        };
        document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, [open, close]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (disabled) return;
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            if (open && focusIndex >= 0) { onChange?.(options[focusIndex].value); close(); }
            else setOpen(true);
        } else if (e.key === "ArrowDown") {
            e.preventDefault();
            if (!open) setOpen(true);
            setFocusIndex(i => Math.min(i + 1, options.length - 1));
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setFocusIndex(i => Math.max(i - 1, 0));
        } else if (e.key === "Escape") { close(); }
    };

    useEffect(() => {
        if (open && focusIndex >= 0 && listRef.current) {
            const el = listRef.current.children[focusIndex] as HTMLElement | undefined;
            el?.scrollIntoView({ block: "nearest" });
        }
    }, [focusIndex, open]);

    return (
        <div ref={containerRef} className="flex flex-col gap-[6px]">
            <label htmlFor={id} className="text-body-sm font-medium text-ink">
                {label}
                {required && <span className="ms-1 text-danger" aria-hidden>*</span>}
            </label>
            <div ref={ref} className="relative">
                <button
                    id={id}
                    type="button"
                    role="combobox"
                    aria-expanded={open}
                    aria-haspopup="listbox"
                    aria-invalid={hasError}
                    disabled={disabled}
                    onClick={() => !disabled && setOpen(o => !o)}
                    onKeyDown={handleKeyDown}
                    className={clsx(
                        "smx-input flex h-11 items-center justify-between rounded-sm pe-3 ps-4 text-start",
                        hasError && "input-error",
                        disabled && "opacity-60 cursor-not-allowed",
                    )}
                >
                    <span className={clsx(!selected && "text-ink-tertiary")}>
                        {selected ? (
                            <>{selected.icon && <span className="me-2">{selected.icon}</span>}{selected.label}</>
                        ) : (placeholder ?? "")}
                    </span>
                    <svg className="h-4 w-4 shrink-0 text-ink-tertiary" data-dir-icon viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                    </svg>
                </button>

                {open && (
                    <ul
                        ref={listRef}
                        role="listbox"
                        className="absolute z-dropdown mt-1 w-full overflow-auto rounded-sm border border-border bg-surface py-1 shadow-md animate-scale-in"
                        style={{ maxHeight: 240 }}
                    >
                        {options.map((opt, i) => (
                            <li
                                key={opt.value}
                                role="option"
                                aria-selected={opt.value === value}
                                className={clsx(
                                    "flex cursor-pointer items-center px-4 py-2 text-body-md transition-colors duration-fast",
                                    opt.value === value && "bg-brand/8 text-brand font-medium",
                                    focusIndex === i && "bg-surface-alt",
                                    opt.value !== value && focusIndex !== i && "hover:bg-surface-alt",
                                )}
                                onClick={() => { onChange?.(opt.value); close(); }}
                            >
                                {opt.icon && <span className="me-2">{opt.icon}</span>}
                                {opt.label}
                            </li>
                        ))}
                    </ul>
                )}
            </div>
            {hasError ? (
                <p className="flex items-center gap-1 text-body-sm text-danger" role="alert">
                    <span aria-hidden>⚠</span> {errorText}
                </p>
            ) : helperText ? (
                <p className="text-body-sm text-ink-tertiary">{helperText}</p>
            ) : null}
        </div>
    );
});
