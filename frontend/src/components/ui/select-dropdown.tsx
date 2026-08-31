"use client";

import { forwardRef, useId, useRef, useState, useEffect, useCallback } from "react";
import clsx from "clsx";

export interface SelectOption {
  value: string;
  label: string;
  icon?: string;
}

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
  ref,
) {
  const autoId = useId();
  const id = providedId ?? autoId;
  const [open, setOpen] = useState(false);
  const [focusIndex, setFocusIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const hasError = Boolean(errorText);
  const selected = options.find((o) => o.value === value);

  const close = useCallback(() => {
    setOpen(false);
    setFocusIndex(-1);
  }, []);

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
      if (open && focusIndex >= 0) {
        onChange?.(options[focusIndex].value);
        close();
      } else {
        setOpen(true);
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!open) setOpen(true);
      setFocusIndex((i) => Math.min(i + 1, options.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocusIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Escape") {
      close();
    }
  };

  useEffect(() => {
    if (open && focusIndex >= 0 && listRef.current) {
      const el = listRef.current.children[focusIndex] as HTMLElement | undefined;
      el?.scrollIntoView({ block: "nearest" });
    }
  }, [focusIndex, open]);

  return (
    <div ref={containerRef} className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-sm font-medium text-ink-secondary">
        {label}
        {required ? <span className="ms-1 text-danger" aria-hidden>*</span> : null}
      </label>
      <div ref={ref} className="relative">
        <button
          id={id}
          type="button"
          role="combobox"
          aria-expanded={open}
          aria-haspopup="listbox"
          aria-controls={open ? `${id}-listbox` : undefined}
          aria-invalid={hasError}
          disabled={disabled}
          onClick={() => !disabled && setOpen((o) => !o)}
          onKeyDown={handleKeyDown}
          className={clsx(
            "smx-input flex h-9 items-center justify-between pe-2.5 ps-2.5 text-start text-sm",
            hasError && "input-error",
            disabled && "cursor-not-allowed opacity-40",
          )}
        >
          <span className={clsx("truncate", !selected && "text-ink-tertiary")}>
            {selected ? (
              <>
                {selected.icon ? <span className="me-2">{selected.icon}</span> : null}
                {selected.label}
              </>
            ) : (
              placeholder ?? ""
            )}
          </span>
          <svg className="h-4 w-4 shrink-0 text-ink-tertiary" data-dir-icon viewBox="0 0 20 20" fill="currentColor" aria-hidden>
            <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
          </svg>
        </button>

        {open ? (
          <ul
            ref={listRef}
            id={`${id}-listbox`}
            role="listbox"
            className="absolute z-dropdown mt-1 max-h-60 w-full overflow-auto rounded-md border border-line bg-surface p-1 shadow-md"
          >
            {options.map((opt, i) => {
              const isSelected = opt.value === value;
              const isFocused = focusIndex === i;
              return (
                <li
                  key={opt.value}
                  role="option"
                  aria-selected={isSelected}
                  className={clsx(
                    "flex min-h-8 cursor-pointer items-center rounded-sm px-2 text-sm transition-colors duration-fast",
                    isSelected
                      ? "border-s-2 border-ink bg-ink/[0.035] ps-1.5 font-medium text-ink"
                      : isFocused
                        ? "bg-ink/[0.045] text-ink"
                        : "text-ink-secondary hover:bg-ink/[0.03] hover:text-ink",
                  )}
                  onClick={() => {
                    onChange?.(opt.value);
                    close();
                  }}
                >
                  {opt.icon ? <span className="me-2 shrink-0">{opt.icon}</span> : null}
                  <span className="truncate">{opt.label}</span>
                </li>
              );
            })}
          </ul>
        ) : null}
      </div>
      {hasError ? (
        <p className="text-xs text-danger" role="alert">{errorText}</p>
      ) : helperText ? (
        <p className="text-xs text-ink-tertiary">{helperText}</p>
      ) : null}
    </div>
  );
});
