"use client"

import * as React from "react"
import {
  IconBrain,
  IconChartBar,
  IconChartCandle,
  IconChartLine,
  IconDashboard,
  IconHelp,
  IconMoodSearch,
  IconNews,
  IconRobot,
  IconSearch,
  IconSettings,
  IconWaveSine,
} from "@tabler/icons-react"

import { NavMain } from '@/components/nav-main'
import { NavSecondary } from '@/components/nav-secondary'
import { NavUser } from '@/components/nav-user'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar'

const data = {
  user: {
    name: "HackEurope 2026",
    email: "team@feelow.ai",
    avatar: "",
  },
  navMain: [
    {
      title: "Dashboard",
      url: "#",
      icon: IconDashboard,
    },
    {
      title: "Agent Pipeline",
      url: "#",
      icon: IconRobot,
    },
    {
      title: "Price & Sentiment",
      url: "#",
      icon: IconChartLine,
    },
    {
      title: "Multi-Model",
      url: "#",
      icon: IconMoodSearch,
    },
    {
      title: "Technicals",
      url: "#",
      icon: IconChartCandle,
    },
    {
      title: "AI Analyst",
      url: "#",
      icon: IconBrain,
    },
    {
      title: "News Feed",
      url: "#",
      icon: IconNews,
    },
  ],
  navSecondary: [
    {
      title: "Settings",
      url: "#",
      icon: IconSettings,
    },
    {
      title: "Help",
      url: "#",
      icon: IconHelp,
    },
    {
      title: "Search",
      url: "#",
      icon: IconSearch,
    },
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            >
              <a href="#">
                <IconWaveSine className="!size-5" />
                <span className="text-base font-semibold">Feelow</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
    </Sidebar>
  )
}
