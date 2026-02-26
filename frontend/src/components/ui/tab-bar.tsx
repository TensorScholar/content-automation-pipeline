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
        <div className={clsx("flex border-b-2 border-border", className)} role="tablist">
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    role="tab"
                    aria-selected={tab.id === activeTab}
                    onClick={() => onChange(tab.id)}
                    className={clsx(
                        "relative px-4 py-3 text-body-md font-medium transition-colors duration-normal",
                        tab.id === activeTab
                            ? "text-brand"
                            : "text-ink-secondary hover:text-ink",
                    )}
                >
                    {tab.label}
                    {tab.count !== undefined && (
                        <span className={clsx(
                            "ms-2 inline-flex h-5 min-w-5 items-center justify-center rounded-full px-1.5 text-body-sm font-medium",
                            tab.id === activeTab ? "bg-brand text-white" : "bg-surface-alt text-ink-secondary",
                        )}>
                            {tab.count}
                        </span>
                    )}
                    {tab.id === activeTab && (
                        <span className="absolute inset-x-0 -bottom-[2px] h-[2px] bg-brand" />
                    )}
                </button>
            ))}
        </div>
    );
}
