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
        <div className={clsx("inline-flex rounded-md border border-black/8 bg-black/[0.03] p-1 dark:border-white/10 dark:bg-white/[0.06]", className)} role="tablist">
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    role="tab"
                    aria-selected={tab.id === activeTab}
                    onClick={() => onChange(tab.id)}
                    className={clsx(
                        "relative rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors duration-normal",
                        tab.id === activeTab
                            ? "bg-white text-ink shadow-[0_1px_2px_rgb(0_0_0/0.06)] dark:bg-white/[0.12] dark:text-white"
                            : "text-ink-secondary hover:text-ink",
                    )}
                >
                    {tab.label}
                    {tab.count !== undefined && (
                        <span className={clsx(
                            "ms-2 inline-flex h-5 min-w-5 items-center justify-center rounded-full px-1.5 text-[11px] font-semibold",
                            tab.id === activeTab ? "bg-brand text-white" : "bg-black/[0.06] text-ink-secondary dark:bg-white/[0.1]",
                        )}>
                            {tab.count}
                        </span>
                    )}
                </button>
            ))}
        </div>
    );
}
