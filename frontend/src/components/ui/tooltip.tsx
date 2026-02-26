"use client";
import { useState, useRef, useEffect } from "react";
import clsx from "clsx";

export interface TooltipProps {
    content: string;
    children: React.ReactNode;
    position?: "top" | "bottom" | "left" | "right";
}

export function Tooltip({ content, children, position = "top" }: TooltipProps) {
    const [show, setShow] = useState(false);
    const timerRef = useRef<number | null>(null);

    const handleEnter = () => { timerRef.current = window.setTimeout(() => setShow(true), 200); };
    const handleLeave = () => { if (timerRef.current) clearTimeout(timerRef.current); setShow(false); };

    useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

    const positionStyles = {
        top: "bottom-full mb-2 start-1/2 -translate-x-1/2",
        bottom: "top-full mt-2 start-1/2 -translate-x-1/2",
        left: "end-full me-2 top-1/2 -translate-y-1/2",
        right: "start-full ms-2 top-1/2 -translate-y-1/2",
    };

    return (
        <span className="relative inline-flex" onMouseEnter={handleEnter} onMouseLeave={handleLeave} onFocus={handleEnter} onBlur={handleLeave}>
            {children}
            {show && (
                <span
                    role="tooltip"
                    className={clsx(
                        "absolute z-tooltip whitespace-nowrap rounded-sm bg-ink px-3 py-1.5 text-body-sm text-white shadow-md animate-fade-in",
                        positionStyles[position],
                    )}
                >
                    {content}
                </span>
            )}
        </span>
    );
}
