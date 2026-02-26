"use client";
import { forwardRef, useId } from "react";
import clsx from "clsx";

export interface InputFieldProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size"> {
    label: string;
    helperText?: string;
    errorText?: string;
    successText?: string;
    showCharCount?: boolean;
    inputSize?: "sm" | "md" | "lg";
}

export const InputField = forwardRef<HTMLInputElement, InputFieldProps>(function InputField(
    { label, helperText, errorText, successText, showCharCount, maxLength, inputSize = "md", className, id: providedId, value, ...rest },
    ref
) {
    const autoId = useId();
    const id = providedId ?? autoId;
    const helperId = `${id}-helper`;
    const errorId = `${id}-error`;
    const hasError = Boolean(errorText);
    const hasSuccess = Boolean(successText) && !hasError;

    const heights = { sm: "h-9", md: "h-11", lg: "h-[52px]" };

    return (
        <div className="flex flex-col gap-[6px]">
            <label htmlFor={id} className="text-body-sm font-medium text-ink">
                {label}
                {rest.required && <span className="ms-1 text-danger" aria-hidden>*</span>}
            </label>
            <input
                ref={ref}
                id={id}
                value={value}
                maxLength={maxLength}
                aria-invalid={hasError}
                aria-required={rest.required}
                aria-describedby={hasError ? errorId : helperText ? helperId : undefined}
                className={clsx(
                    "smx-input rounded-sm",
                    heights[inputSize],
                    hasError && "input-error",
                    hasSuccess && "input-success",
                    className,
                )}
                {...rest}
            />
            <div className="flex items-center justify-between">
                {hasError ? (
                    <p id={errorId} className="flex items-center gap-1 text-body-sm text-danger" role="alert">
                        <span aria-hidden>⚠</span> {errorText}
                    </p>
                ) : hasSuccess ? (
                    <p className="text-body-sm text-success">{successText}</p>
                ) : helperText ? (
                    <p id={helperId} className="text-body-sm text-ink-tertiary">{helperText}</p>
                ) : <span />}
                {showCharCount && maxLength && (
                    <span className={clsx("text-body-sm", (String(value ?? "").length) >= maxLength * 0.8 ? "text-warning" : "text-ink-tertiary")}>
                        {String(value ?? "").length}/{maxLength}
                    </span>
                )}
            </div>
        </div>
    );
});
