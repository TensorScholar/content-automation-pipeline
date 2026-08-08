"use client";
import clsx from "clsx";

export interface TabItem {
    id: string;
    label: string;
    count?: number;
}

export interface TabBarProps {
    tabs: TabItem[];
    activeTab: string;
    onChange: (tabId: string) => void;
    className?: string;
}

export function TabBar({ tabs, activeTab, onChange, className }: TabBarProps) {
    return (
        <div className={clsx("inline-flex max-w-full items-center gap-1 overflow-x-auto rounded-[13px] border border-black/[0.055] bg-black/[0.035] p-1 dark:border-white/[0.075] dark:bg-white/[0.055]", className)} role="tablist">
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={tab.id === activeTab}
                    onClick={() => onChange(tab.id)}
                    className={clsx(
                        "inline-flex min-h-[34px] shrink-0 items-center rounded-[10px] px-3 text-[12px] font-semibold transition-[background-color,color,box-shadow] duration-150 focus-visible:outline-none",
                        tab.id === activeTab
                            ? "bg-white text-ink shadow-[0_1px_2px_rgb(0_0_0/0.06)] dark:bg-white/[0.12] dark:text-white"
                            : "text-ink-secondary hover:text-ink",
                    )}
                >
                    {tab.label}
                    {tab.count !== undefined && (
                        <span className={clsx(
                            "ms-2 inline-flex h-5 min-w-5 items-center justify-center rounded-full px-1.5 text-[10px] tabular-nums",
                            tab.id === activeTab ? "bg-brand/10 text-brand" : "bg-black/[0.06] text-ink-secondary dark:bg-white/[0.1]",
                        )}>
                            {tab.count}
                        </span>
                    )}
                </button>
            ))}
        </div>
    );
}
