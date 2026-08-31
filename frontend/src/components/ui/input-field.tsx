"use client";

import { forwardRef, useId } from "react";
import clsx from "clsx";

export interface InputFieldProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size" | "prefix"> {
  label?: string;
  error?: boolean;
  errorText?: string;
  successText?: string;
  helperText?: React.ReactNode;
  prefix?: React.ReactNode;
  suffix?: React.ReactNode;
  fullWidth?: boolean;
  showCharCount?: boolean;
  inputSize?: "sm" | "md" | "lg";
}

export const InputField = forwardRef<HTMLInputElement, InputFieldProps>(function InputField(
  { label, error = false, helperText, showCharCount, errorText, successText, maxLength, inputSize = "md", className, id: providedId, value, prefix, suffix, fullWidth, ...rest },
  ref,
) {
  const autoId = useId();
  const id = providedId ?? autoId;
  const helperId = `${id}-helper`;
  const errorId = `${id}-error`;
  const hasError = error || Boolean(errorText);
  const hasSuccess = Boolean(successText) && !hasError;
  const heights = { sm: "h-8", md: "h-9", lg: "h-9" };

  return (
    <div className={clsx("flex flex-col gap-1.5", fullWidth && "w-full")}>
      {label ? (
        <label htmlFor={id} className="text-sm font-medium text-ink-secondary">
          {label}{rest.required ? <span className="ms-1 text-danger" aria-hidden>*</span> : null}
        </label>
      ) : null}
      <div className="relative">
        {prefix ? <span className="pointer-events-none absolute inset-y-0 start-2.5 inline-flex items-center text-ink-tertiary">{prefix}</span> : null}
        <input
          ref={ref}
          id={id}
          value={value}
          maxLength={maxLength}
          aria-invalid={hasError}
          aria-required={rest.required}
          aria-describedby={hasError ? errorId : helperText ? helperId : undefined}
          className={clsx(
            "smx-input text-sm",
            heights[inputSize],
            prefix && "ps-8",
            suffix && "pe-8",
            hasError && "input-error",
            hasSuccess && "input-success",
            className,
          )}
          {...rest}
        />
        {suffix ? <span className="pointer-events-none absolute inset-y-0 end-2.5 inline-flex items-center text-ink-tertiary">{suffix}</span> : null}
      </div>
      {(hasError || hasSuccess || helperText || (showCharCount && maxLength)) ? (
        <div className="flex min-h-[16px] items-start justify-between gap-3">
          {hasError ? (
            <p id={errorId} className="text-xs text-danger" role="alert">{errorText}</p>
          ) : hasSuccess ? (
            <p className="text-xs text-success">{successText}</p>
          ) : helperText ? (
            <p id={helperId} className="text-xs text-ink-tertiary">{helperText}</p>
          ) : <span />}
          {showCharCount && maxLength ? (
            <span className={clsx("shrink-0 text-xs tabular-nums", String(value ?? "").length >= maxLength * 0.8 ? "text-warning" : "text-ink-tertiary")}>
              {String(value ?? "").length}/{maxLength}
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
});
