"use client"

import * as React from "react"
import {
  closestCenter,
  DndContext,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
  type UniqueIdentifier,
} from "@dnd-kit/core"
import { restrictToVerticalAxis } from "@dnd-kit/modifiers"
import {
  arrayMove,
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import {
  IconChevronDown,
  IconChevronLeft,
  IconChevronRight,
  IconChevronsLeft,
  IconChevronsRight,
  IconDotsVertical,
  IconGripVertical,
  IconLayoutColumns,
} from "@tabler/icons-react"
import {
  flexRender,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type ColumnFiltersState,
  type Row,
  type SortingState,
  type VisibilityState,
} from "@tanstack/react-table"
import { z } from "zod"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { useTicker } from "@/lib/ticker-context"

/* ─────────────────────── Schema ─────────────────────── */
export const schema = z.object({
  id: z.number(),
  header: z.string(),
  type: z.string(),
  sentiment: z.string(),
  confidence: z.string(),
  published: z.string(),
})

type FeedRow = z.infer<typeof schema>

/* ─────────────── Fake Polymarket bets ─────────────── */
const POLYMARKET_BETS: FeedRow[] = [
  {
    id: 1001,
    header: "Will AAPL close above $270 by March 2026?",
    type: "Polymarket",
    sentiment: "Positive",
    confidence: "72%",
    published: "2025-06-22 09:15",
  },
  {
    id: 1002,
    header: "Tesla stock drops 10%+ in February?",
    type: "Polymarket",
    sentiment: "Negative",
    confidence: "58%",
    published: "2025-06-21 14:30",
  },
  {
    id: 1003,
    header: "NVIDIA beats Q4 earnings consensus?",
    type: "Polymarket",
    sentiment: "Positive",
    confidence: "84%",
    published: "2025-06-20 11:00",
  },
  {
    id: 1004,
    header: "Fed rate cut before April 2026?",
    type: "Polymarket",
    sentiment: "Neutral",
    confidence: "45%",
    published: "2025-06-19 18:45",
  },
  {
    id: 1005,
    header: "Bitcoin above $100K by end of Q1?",
    type: "Polymarket",
    sentiment: "Positive",
    confidence: "63%",
    published: "2025-06-18 22:10",
  },
  {
    id: 1006,
    header: "Google announces stock buyback Q1 2026?",
    type: "Polymarket",
    sentiment: "Positive",
    confidence: "51%",
    published: "2025-06-17 08:30",
  },
  {
    id: 1007,
    header: "S&P 500 correction >5% in March?",
    type: "Polymarket",
    sentiment: "Negative",
    confidence: "37%",
    published: "2025-06-16 16:00",
  },
  {
    id: 1008,
    header: "Meta Platforms revenue beats $42B forecast?",
    type: "Polymarket",
    sentiment: "Positive",
    confidence: "69%",
    published: "2025-06-15 10:20",
  },
]

/* ─────────────── Sentiment badge ─────────────── */
function SentimentBadge({ value }: { value: string }) {
  const s = value.toLowerCase()
  const cls =
    s === "positive"
      ? "border-green-500/40 text-green-400 bg-green-500/10"
      : s === "negative"
        ? "border-red-500/40 text-red-400 bg-red-500/10"
        : "border-yellow-500/40 text-yellow-400 bg-yellow-500/10"
  return (
    <Badge variant="outline" className={`px-1.5 ${cls}`}>
      {value}
    </Badge>
  )
}

/* ──────────────── Drag handle ──────────────── */
function DragHandle({ id }: { id: number }) {
  const { attributes, listeners } = useSortable({ id })
  return (
    <Button
      {...attributes}
      {...listeners}
      variant="ghost"
      size="icon"
      className="text-muted-foreground size-7 hover:bg-transparent"
    >
      <IconGripVertical className="text-muted-foreground size-3" />
      <span className="sr-only">Drag to reorder</span>
    </Button>
  )
}

/* ──────────────── Columns ──────────────── */
const columns: ColumnDef<FeedRow>[] = [
  {
    id: "drag",
    header: () => null,
    cell: ({ row }) => <DragHandle id={row.original.id} />,
  },
  {
    id: "select",
    header: ({ table }) => (
      <div className="flex items-center justify-center">
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && "indeterminate")
          }
          onCheckedChange={(v) => table.toggleAllPageRowsSelected(!!v)}
          aria-label="Select all"
        />
      </div>
    ),
    cell: ({ row }) => (
      <div className="flex items-center justify-center">
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(v) => row.toggleSelected(!!v)}
          aria-label="Select row"
        />
      </div>
    ),
    enableSorting: false,
    enableHiding: false,
  },
  {
    accessorKey: "header",
    header: "Headline",
    cell: ({ row }) => (
      <div className="max-w-[340px] truncate font-medium">
        {row.original.header}
      </div>
    ),
    enableHiding: false,
  },
  {
    accessorKey: "type",
    header: "Source",
    cell: ({ row }) => (
      <div className="w-32">
        <Badge variant="outline" className="text-muted-foreground px-1.5">
          {row.original.type}
        </Badge>
      </div>
    ),
  },
  {
    accessorKey: "sentiment",
    header: "Sentiment",
    cell: ({ row }) => <SentimentBadge value={row.original.sentiment} />,
  },
  {
    accessorKey: "confidence",
    header: () => <div className="text-right">Confidence</div>,
    cell: ({ row }) => (
      <div className="text-right font-mono text-sm tabular-nums">
        {row.original.confidence}
      </div>
    ),
  },
  {
    accessorKey: "published",
    header: "Published",
    cell: ({ row }) => {
      const d = new Date(row.original.published)
      const valid = !isNaN(d.getTime())
      return (
        <div className="text-muted-foreground text-sm whitespace-nowrap">
          {valid
            ? `${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })} ${d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}`
            : row.original.published}
        </div>
      )
    },
  },
  {
    id: "actions",
    cell: () => (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            className="data-[state=open]:bg-muted text-muted-foreground flex size-8"
            size="icon"
          >
            <IconDotsVertical />
            <span className="sr-only">Open menu</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-32">
          <DropdownMenuItem>View source</DropdownMenuItem>
          <DropdownMenuItem>Copy headline</DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem variant="destructive">Hide</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    ),
  },
]

/* ──────────── Draggable row ──────────── */
function DraggableRow({ row }: { row: Row<FeedRow> }) {
  const { transform, transition, setNodeRef, isDragging } = useSortable({
    id: row.original.id,
  })
  return (
    <TableRow
      data-state={row.getIsSelected() && "selected"}
      data-dragging={isDragging}
      ref={setNodeRef}
      className="relative z-0 data-[dragging=true]:z-10 data-[dragging=true]:opacity-80"
      style={{
        transform: CSS.Transform.toString(transform),
        transition: transition,
      }}
    >
      {row.getVisibleCells().map((cell) => (
        <TableCell key={cell.id}>
          {flexRender(cell.column.columnDef.cell, cell.getContext())}
        </TableCell>
      ))}
    </TableRow>
  )
}

/* ──────────── Inner table (reused per tab) ──────────── */
function FeedTable({ data: initialData }: { data: FeedRow[] }) {
  const [data, setData] = React.useState(initialData)
  const [rowSelection, setRowSelection] = React.useState({})
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({})
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  )
  const [sorting, setSorting] = React.useState<SortingState>([])
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 10,
  })

  const sortableId = React.useId()
  const sensors = useSensors(
    useSensor(MouseSensor, {}),
    useSensor(TouchSensor, {}),
    useSensor(KeyboardSensor, {})
  )

  React.useEffect(() => {
    setData(initialData)
    setPagination((p) => ({ ...p, pageIndex: 0 }))
  }, [initialData])

  const dataIds = React.useMemo<UniqueIdentifier[]>(
    () => data.map(({ id }) => id),
    [data]
  )

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
      columnFilters,
      pagination,
    },
    getRowId: (row) => row.id.toString(),
    enableRowSelection: true,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
  })

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event
    if (active && over && active.id !== over.id) {
      setData((prev) => {
        const oldIndex = dataIds.indexOf(active.id)
        const newIndex = dataIds.indexOf(over.id)
        return arrayMove(prev, oldIndex, newIndex)
      })
    }
  }

  return (
    <>
      <div className="overflow-hidden rounded-lg border">
        <DndContext
          collisionDetection={closestCenter}
          modifiers={[restrictToVerticalAxis]}
          onDragEnd={handleDragEnd}
          sensors={sensors}
          id={sortableId}
        >
          <Table>
            <TableHeader className="bg-muted sticky top-0 z-10">
              {table.getHeaderGroups().map((hg) => (
                <TableRow key={hg.id}>
                  {hg.headers.map((h) => (
                    <TableHead key={h.id} colSpan={h.colSpan}>
                      {h.isPlaceholder
                        ? null
                        : flexRender(h.column.columnDef.header, h.getContext())}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody className="**:data-[slot=table-cell]:first:w-8">
              {table.getRowModel().rows?.length ? (
                <SortableContext
                  items={dataIds}
                  strategy={verticalListSortingStrategy}
                >
                  {table.getRowModel().rows.map((row) => (
                    <DraggableRow key={row.id} row={row} />
                  ))}
                </SortableContext>
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={columns.length}
                    className="h-24 text-center"
                  >
                    No results.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </DndContext>
      </div>

      {/* pagination */}
      <div className="flex items-center justify-between px-4">
        <div className="text-muted-foreground hidden flex-1 text-sm lg:flex">
          {table.getFilteredSelectedRowModel().rows.length} of{" "}
          {table.getFilteredRowModel().rows.length} row(s) selected.
        </div>
        <div className="flex w-full items-center gap-8 lg:w-fit">
          <div className="hidden items-center gap-2 lg:flex">
            <Label htmlFor="rows-per-page" className="text-sm font-medium">
              Rows per page
            </Label>
            <Select
              value={`${table.getState().pagination.pageSize}`}
              onValueChange={(v) => table.setPageSize(Number(v))}
            >
              <SelectTrigger className="w-20" id="rows-per-page">
                <SelectValue
                  placeholder={table.getState().pagination.pageSize}
                />
              </SelectTrigger>
              <SelectContent side="top">
                {[10, 20, 30, 40, 50].map((ps) => (
                  <SelectItem key={ps} value={`${ps}`}>
                    {ps}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex w-fit items-center justify-center text-sm font-medium">
            Page {table.getState().pagination.pageIndex + 1} of{" "}
            {table.getPageCount()}
          </div>
          <div className="ml-auto flex items-center gap-2 lg:ml-0">
            <Button
              variant="outline"
              className="hidden h-8 w-8 p-0 lg:flex"
              onClick={() => table.setPageIndex(0)}
              disabled={!table.getCanPreviousPage()}
            >
              <span className="sr-only">Go to first page</span>
              <IconChevronsLeft />
            </Button>
            <Button
              variant="outline"
              className="size-8"
              size="icon"
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
            >
              <span className="sr-only">Go to previous page</span>
              <IconChevronLeft />
            </Button>
            <Button
              variant="outline"
              className="size-8"
              size="icon"
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
            >
              <span className="sr-only">Go to next page</span>
              <IconChevronRight />
            </Button>
            <Button
              variant="outline"
              className="hidden size-8 lg:flex"
              size="icon"
              onClick={() => table.setPageIndex(table.getPageCount() - 1)}
              disabled={!table.getCanNextPage()}
            >
              <span className="sr-only">Go to last page</span>
              <IconChevronsRight />
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}

/* ─────────── helpers for fake sentiment on articles ─────────── */
const SENTIMENTS = ["Positive", "Negative", "Neutral"] as const
function fakeSentiment(i: number): string {
  return SENTIMENTS[i % SENTIMENTS.length]
}
function fakeConfidence(i: number): string {
  const vals = [87, 92, 73, 65, 81, 94, 56, 78, 68, 91, 83, 76, 62, 88, 95, 71, 59, 86, 74, 80]
  return `${vals[i % vals.length]}%`
}

/* ════════════════════════════════════════════════════════════════
   Main export — self-fetching, no data prop needed
   ════════════════════════════════════════════════════════════════ */
export function DataTable() {
  const { ticker } = useTicker()
  const [articles, setArticles] = React.useState<FeedRow[]>([])
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    setLoading(true)
    fetch(`http://localhost:8000/api/news?ticker=${ticker}`)
      .then((r) => r.json())
      .then(
        (
          raw: {
            title: string
            source: string
            published: string
            link: string
            sentiment?: string
            confidence?: number
          }[]
        ) => {
          setArticles(
            raw.map((item, i) => ({
              id: 2000 + i,
              header: item.title,
              type: item.source || "article",
              sentiment: item.sentiment
                ? item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)
                : "Neutral",
              confidence: item.confidence
                ? `${Math.round(item.confidence * 100)}%`
                : "—",
              published: item.published,
            }))
          )
        }
      )
      .catch(() => setArticles([]))
      .finally(() => setLoading(false))
  }, [ticker])

  const allData = React.useMemo(
    () =>
      [...POLYMARKET_BETS, ...articles].sort(
        (a, b) =>
          new Date(b.published).getTime() - new Date(a.published).getTime()
      ),
    [articles]
  )

  return (
    <Tabs defaultValue="all" className="w-full flex-col justify-start gap-6">
      <div className="flex items-center justify-between px-4 lg:px-6">
        <Label htmlFor="view-selector" className="sr-only">
          View
        </Label>
        <Select defaultValue="all">
          <SelectTrigger
            className="flex w-fit @4xl/main:hidden"
            id="view-selector"
          >
            <SelectValue placeholder="Select a view" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">ALL</SelectItem>
            <SelectItem value="polymarket-bets">Polymarket bets</SelectItem>
            <SelectItem value="article">article</SelectItem>
          </SelectContent>
        </Select>
        <TabsList className="**:data-[slot=badge]:bg-muted-foreground/30 hidden **:data-[slot=badge]:size-5 **:data-[slot=badge]:rounded-full **:data-[slot=badge]:px-1 @4xl/main:flex">
          <TabsTrigger value="all">ALL</TabsTrigger>
          <TabsTrigger value="polymarket-bets">
            Polymarket bets <Badge variant="secondary">{POLYMARKET_BETS.length}</Badge>
          </TabsTrigger>
          <TabsTrigger value="article">
            article <Badge variant="secondary">{articles.length}</Badge>
          </TabsTrigger>
        </TabsList>
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <IconLayoutColumns />
                <span className="hidden lg:inline">Customize Columns</span>
                <span className="lg:hidden">Columns</span>
                <IconChevronDown />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              {["type", "sentiment", "confidence", "published"].map((key) => (
                <DropdownMenuCheckboxItem
                  key={key}
                  className="capitalize"
                  checked
                >
                  {key}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* ALL */}
      <TabsContent
        value="all"
        className="relative flex flex-col gap-4 overflow-auto px-4 lg:px-6"
      >
        {loading ? (
          <div className="flex h-40 items-center justify-center gap-2">
            <span className="text-muted-foreground text-sm">
              Loading news feed…
            </span>
          </div>
        ) : (
          <FeedTable data={allData} />
        )}
      </TabsContent>

      {/* Polymarket bets */}
      <TabsContent
        value="polymarket-bets"
        className="relative flex flex-col gap-4 overflow-auto px-4 lg:px-6"
      >
        <FeedTable data={POLYMARKET_BETS} />
      </TabsContent>

      {/* article */}
      <TabsContent
        value="article"
        className="relative flex flex-col gap-4 overflow-auto px-4 lg:px-6"
      >
        {loading ? (
          <div className="flex h-40 items-center justify-center gap-2">
            <span className="text-muted-foreground text-sm">
              Loading articles…
            </span>
          </div>
        ) : (
          <FeedTable data={articles} />
        )}
      </TabsContent>
    </Tabs>
  )
}
