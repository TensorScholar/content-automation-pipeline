"use client";
import { forwardRef, useId } from "react";
import clsx from "clsx";

export interface InputFieldProps
    extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size" | "prefix"> {
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
    { label, helperText, showCharCount, errorText, successText, maxLength, inputSize = "md", className, id: providedId, value, prefix, suffix, fullWidth, ...rest },
    ref
) {
    const autoId = useId();
    const id = providedId ?? autoId;
    const helperId = `${id}-helper`;
    const errorId = `${id}-error`;
    const hasError = Boolean(errorText);
    const hasSuccess = Boolean(successText) && !hasError;

    const heights = { sm: "h-10", md: "h-11", lg: "h-11" };

    return (
        <div className={clsx("flex flex-col gap-1", fullWidth && "w-full")}>
            {label ? (
                <label htmlFor={id} className="text-[12px] font-medium tracking-normal text-ink-secondary">
                    {label}
                    {rest.required && <span className="ms-1 text-danger" aria-hidden>*</span>}
                </label>
            ) : null}

            <div className="relative">
                {prefix ? (
                    <span className="pointer-events-none absolute inset-y-0 start-3 inline-flex items-center text-ink-tertiary">
                        {prefix}
                    </span>
                ) : null}

                <input
                    ref={ref}
                    id={id}
                    value={value}
                    maxLength={maxLength}
                    aria-invalid={hasError}
                    aria-required={rest.required}
                    aria-describedby={hasError ? errorId : helperText ? helperId : undefined}
                    className={clsx(
                        "smx-input rounded-md",
                        heights[inputSize],
                        prefix && "ps-10",
                        suffix && "pe-10",
                        hasError && "input-error",
                        hasSuccess && "input-success",
                        className,
                    )}
                    {...rest}
                />

                {suffix ? (
                    <span className="pointer-events-none absolute inset-y-0 end-3 inline-flex items-center text-ink-tertiary">
                        {suffix}
                    </span>
                ) : null}
            </div>

            <div className="flex items-center justify-between">
                {hasError ? (
                    <p id={errorId} className="flex items-center gap-1 text-[11px] text-danger" role="alert">
                        <span aria-hidden>⚠</span> {errorText}
                    </p>
                ) : hasSuccess ? (
                    <p className="text-[11px] text-success">{successText}</p>
                ) : helperText ? (
                    <p id={helperId} className="text-[11px] text-ink-tertiary">{helperText}</p>
                ) : <span />}
                {showCharCount && maxLength && (
                    <span className={clsx("text-[11px]", (String(value ?? "").length) >= maxLength * 0.8 ? "text-warning" : "text-ink-tertiary")}>
                        {String(value ?? "").length}/{maxLength}
                    </span>
                )}
            </div>
        </div>
    );
});
