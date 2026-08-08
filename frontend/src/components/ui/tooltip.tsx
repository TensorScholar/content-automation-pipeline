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
        top: "bottom-full mb-2 start-1/2 -translate-x-1/2 rtl:translate-x-1/2",
        bottom: "top-full mt-2 start-1/2 -translate-x-1/2 rtl:translate-x-1/2",
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
                        "absolute z-tooltip max-w-[240px] whitespace-normal rounded-[9px] bg-[#1f2227] px-2.5 py-1.5 text-[11px] font-medium leading-4 text-white shadow-[0_12px_30px_-16px_rgb(0_0_0/0.65)] animate-fade-in",
                        positionStyles[position],
                    )}
                >
                    {content}
                </span>
            )}
        </span>
    );
}
