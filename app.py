import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CPUScheduler:
    def __init__(self):
        self.processes = []  # [[CBT1, AT1], [CBT2, AT2], ...]
        self.context_switch_time = 0
        self.time_quantum = 5

    def fcfs(self):
        n = len(self.processes)
        CBT = [p[0] for p in self.processes]
        AT = [p[1] for p in self.processes]

        WT = [0] * n
        TT = [0] * n
        RT = [0] * n
        gantt = []

        sorted_processes = sorted(enumerate(self.processes), key=lambda x: x[1][1])
        time = sorted_processes[0][1][1]

        for idx, (i, (cbt, at)) in enumerate(sorted_processes):
            if time < at:
                time = at

            WT[i] = time - at
            RT[i] = WT[i]

            gantt.append((at, at, i, "arrival"))
            gantt.append((time, time + cbt, i, "execution"))
            time += cbt

            TT[i] = WT[i] + cbt

            # context switch (not after last executed)
            last_process = (idx == len(sorted_processes) - 1)
            if (not last_process) and self.context_switch_time > 0:
                gantt.append((time, time + self.context_switch_time, -1, "context_switch"))
                time += self.context_switch_time

        return gantt, WT, TT, RT

    def spn(self):
        n = len(self.processes)
        CBT = [p[0] for p in self.processes]
        AT = [p[1] for p in self.processes]

        WT = [0] * n
        TT = [0] * n
        RT = [0] * n
        gantt = []

        completed = [False] * n
        time = min(AT)

        while True:
            available = [(i, CBT[i]) for i in range(n) if (not completed[i] and AT[i] <= time)]

            if not available:
                time = min(AT[i] for i in range(n) if not completed[i])
                continue

            current = min(available, key=lambda x: x[1])[0]

            gantt.append((AT[current], AT[current], current, "arrival"))
            gantt.append((time, time + CBT[current], current, "execution"))

            WT[current] = time - AT[current]
            RT[current] = WT[current]
            TT[current] = WT[current] + CBT[current]

            time += CBT[current]
            completed[current] = True

            if all(completed):
                break

            if self.context_switch_time > 0:
                gantt.append((time, time + self.context_switch_time, -1, "context_switch"))
                time += self.context_switch_time

        return gantt, WT, TT, RT

    def hrrn(self):
        n = len(self.processes)
        CBT = [p[0] for p in self.processes]
        AT = [p[1] for p in self.processes]

        WT = [0] * n
        TT = [0] * n
        RT = [0] * n
        gantt = []

        completed = [False] * n
        time = min(AT)

        while True:
            available = []
            for i in range(n):
                if not completed[i] and AT[i] <= time:
                    wait_time = time - AT[i]
                    response_ratio = (wait_time + CBT[i]) / CBT[i]
                    available.append((i, response_ratio))

            if not available:
                time = min(AT[i] for i in range(n) if not completed[i])
                continue

            current = max(available, key=lambda x: x[1])[0]

            gantt.append((AT[current], AT[current], current, "arrival"))
            gantt.append((time, time + CBT[current], current, "execution"))

            WT[current] = time - AT[current]
            RT[current] = WT[current]
            TT[current] = WT[current] + CBT[current]

            time += CBT[current]
            completed[current] = True

            if all(completed):
                break

            if self.context_switch_time > 0:
                gantt.append((time, time + self.context_switch_time, -1, "context_switch"))
                time += self.context_switch_time

        return gantt, WT, TT, RT

    def rr(self):
        n = len(self.processes)
        CBT = [p[0] for p in self.processes]
        AT = [p[1] for p in self.processes]

        WT = [0] * n
        TT = [0] * n
        RT = [-1] * n
        gantt = []

        remaining = CBT.copy()
        time = min(AT)
        ready_queue = []

        while True:
            for i in range(n):
                if AT[i] <= time and remaining[i] > 0 and i not in ready_queue:
                    ready_queue.append(i)

            if not ready_queue:
                if all(remaining[i] == 0 for i in range(n)):
                    break
                time = min(AT[i] for i in range(n) if remaining[i] > 0)
                continue

            # pick next
            current = ready_queue.pop(0)

            if remaining[current] == CBT[current]:
                gantt.append((AT[current], AT[current], current, "arrival"))

            if RT[current] == -1:
                RT[current] = time - AT[current]

            execute_time = min(self.time_quantum, remaining[current])
            gantt.append((time, time + execute_time, current, "execution"))

            # arrivals during execution
            for t in range(time, time + execute_time + 1):
                for i in range(n):
                    if AT[i] == t and remaining[i] > 0 and i != current and i not in ready_queue:
                        ready_queue.append(i)

            remaining[current] -= execute_time
            time += execute_time

            # arrivals during context switch window BEFORE requeueing
            for t in range(time, time + self.context_switch_time + 1):
                for i in range(n):
                    if AT[i] == t and remaining[i] > 0 and i != current and i not in ready_queue:
                        ready_queue.append(i)

            if remaining[current] > 0:
                ready_queue.append(current)
            else:
                TT[current] = time - AT[current]
                WT[current] = TT[current] - CBT[current]

            if self.context_switch_time > 0 and ready_queue:
                gantt.append((time, time + self.context_switch_time, -1, "context_switch"))
                time += self.context_switch_time

        return gantt, WT, TT, RT

    def srtf(self):
        n = len(self.processes)
        CBT = [p[0] for p in self.processes]
        AT = [p[1] for p in self.processes]

        remaining = CBT.copy()
        WT = [0] * n
        TT = [0] * n
        RT = [-1] * n
        gantt = []

        time = min(AT)
        completed = 0
        current_process = None
        execution_start = None
        first_process = True

        while completed < n:
            # choose process with minimum remaining
            min_remaining = float('inf')
            next_process = None

            for i in range(n):
                if AT[i] <= time and remaining[i] > 0:
                    if remaining[i] < min_remaining:
                        min_remaining = remaining[i]
                        next_process = i

            if next_process is None:
                time = min(AT[i] for i in range(n) if remaining[i] > 0)
                continue

            if remaining[next_process] == CBT[next_process]:
                gantt.append((AT[next_process], AT[next_process], next_process, "arrival"))

            if next_process != current_process:
                if current_process is not None and execution_start is not None:
                    gantt.append((execution_start, time, current_process, "execution"))

                if (not first_process) and self.context_switch_time > 0:
                    gantt.append((time, time + self.context_switch_time, -1, "context_switch"))
                    time += self.context_switch_time

                current_process = next_process
                execution_start = time

                if RT[current_process] == -1:
                    RT[current_process] = time - AT[current_process]

            remaining[current_process] -= 1

            if remaining[current_process] == 0:
                gantt.append((execution_start, time + 1, current_process, "execution"))
                completed += 1
                TT[current_process] = time + 1 - AT[current_process]
                WT[current_process] = TT[current_process] - CBT[current_process]
                current_process = None
                execution_start = None

            first_process = False
            time += 1

        if current_process is not None and execution_start is not None:
            gantt.append((execution_start, time, current_process, "execution"))

        return gantt, WT, TT, RT

    def dart(self):
        """
        DART = Dynamic Aging + Response-time RR + burst prediction
        Non-preemptive within quantum, re-evaluates priority at each dispatch.
        Uses RR quantum but chooses next process by dynamic score.
        """

        n = len(self.processes)
        CBT = [p[0] for p in self.processes]
        AT = [p[1] for p in self.processes]

        remaining = CBT.copy()

        WT = [0] * n
        TT = [0] * n
        RT = [-1] * n
        gantt = []

        # Parameters
        q = self.time_quantum
        alpha = 0.5
        A = 10.0
        I_threshold = 2
        w1, w2, w3, w4 = 2.0, 1.0, 3.0, 1.0

        # State
        tau = [float(CBT[i]) for i in range(n)]
        last_ready_time = [AT[i] for i in range(n)]
        last_cpu_burst = [0] * n
        total_cpu_used = [0] * n

        arrived_marked = [False] * n
        completion_time = [-1] * n  # FIX: store completion to compute TT/WT safely

        time = min(AT)
        ready = []

        def add_arrivals_at(t_now, exclude=None):
            for i in range(n):
                if i == exclude:
                    continue
                if AT[i] <= t_now and remaining[i] > 0 and i not in ready:
                    ready.append(i)
                    if not arrived_marked[i]:
                        gantt.append((AT[i], AT[i], i, "arrival"))
                        arrived_marked[i] = True
                    if last_ready_time[i] < AT[i]:
                        last_ready_time[i] = AT[i]

        def score(i):
            wait = time - last_ready_time[i]
            aging = (wait / A) if A > 0 else 0.0
            interactive = 1.0 if last_cpu_burst[i] <= I_threshold else 0.0
            shortness = 1.0 / (tau[i] + 1.0)
            penalty = total_cpu_used[i] / (time + 1.0)
            return (w1 * aging) + (w2 * interactive) + (w3 * shortness) - (w4 * penalty)

        add_arrivals_at(time)

        while True:
            if all(remaining[i] == 0 for i in range(n)):
                break

            if not ready:
                next_t = min(AT[i] for i in range(n) if remaining[i] > 0)
                time = max(time, next_t)
                add_arrivals_at(time)
                if not ready:
                    continue

            ready.sort(key=lambda i: (-score(i), AT[i], i))
            current = ready.pop(0)

            if RT[current] == -1:
                RT[current] = time - AT[current]

            exec_time = min(q, remaining[current])
            start = time
            end = time + exec_time
            gantt.append((start, end, current, "execution"))

            # arrivals during execution
            for t in range(start, end + 1):
                for i in range(n):
                    if AT[i] == t and remaining[i] > 0 and i != current and i not in ready:
                        ready.append(i)
                        if not arrived_marked[i]:
                            gantt.append((AT[i], AT[i], i, "arrival"))
                            arrived_marked[i] = True

            # update stats
            remaining[current] -= exec_time
            total_cpu_used[current] += exec_time
            last_cpu_burst[current] = exec_time

            # burst prediction
            tau[current] = alpha * exec_time + (1 - alpha) * tau[current]

            time = end

            if remaining[current] > 0:
                last_ready_time[current] = time
                ready.append(current)
            else:
                completion_time[current] = time  # FIX

            # context switch block
            if self.context_switch_time > 0 and ready:
                cs_start = time
                cs_end = time + self.context_switch_time

                for t in range(cs_start, cs_end + 1):
                    for i in range(n):
                        if AT[i] == t and remaining[i] > 0 and i not in ready:
                            ready.append(i)
                            if not arrived_marked[i]:
                                gantt.append((AT[i], AT[i], i, "arrival"))
                                arrived_marked[i] = True

                gantt.append((cs_start, cs_end, -1, "context_switch"))
                time = cs_end

        # ---- FINAL METRICS FIX (prevents negative values) ----
        for i in range(n):
            if completion_time[i] == -1:
                completion_time[i] = time
            TT[i] = completion_time[i] - AT[i]
            WT[i] = TT[i] - CBT[i]
            if RT[i] < 0:
                RT[i] = 0
            if TT[i] < 0:
                TT[i] = 0
            if WT[i] < 0:
                WT[i] = 0

        return gantt, WT, TT, RT


class SchedulerApp:
    def __init__(self, root):
        self.scheduler = CPUScheduler()
        self.root = root
        self.root.title("CPU Scheduling Simulator")
        self.create_widgets()

    def create_widgets(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=1)

        input_frame = tk.Frame(self.root)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        input_frame.grid_rowconfigure(6, weight=1)

        tk.Label(input_frame, text="Processes (CBT, AT) as List:").grid(row=0, column=0, sticky="w")
        self.process_input = tk.Text(input_frame, width=30, height=5)
        self.process_input.grid(row=1, column=0, pady=5)
        self.process_input.insert("1.0", "[[6, 1], [10, 7], [2, 9], [3, 25]]")

        tk.Label(input_frame, text="Context Switch Time:").grid(row=2, column=0, sticky="w")
        self.cs_input = tk.Entry(input_frame)
        self.cs_input.grid(row=3, column=0, pady=5)
        self.cs_input.insert(0, "2")

        tk.Label(input_frame, text="Time Quantum:").grid(row=4, column=0, sticky="w")
        self.tq_input = tk.Entry(input_frame)
        self.tq_input.grid(row=5, column=0, pady=5)
        self.tq_input.insert(0, "5")

        self.run_button = tk.Button(input_frame, text="Run", command=self.run_simulation)
        self.run_button.grid(row=6, column=0, pady=10, sticky="s")

        scrollable_frame = tk.Frame(self.root)
        scrollable_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        canvas = tk.Canvas(scrollable_frame)
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        canvas.configure(yscrollcommand=scrollbar.set)

        self.output_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.output_frame, anchor="nw")
        self.output_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(0, weight=1)

    def create_metrics_table(self, parent, title, data):
        table_frame = ttk.LabelFrame(parent, text=title, padding="10")
        table_frame.grid(sticky="nsew", padx=5, pady=5)

        headers = ['Process', 'Wait Time', 'Turnaround Time', 'Response Time']
        for col, header in enumerate(headers):
            ttk.Label(table_frame, text=header, font=('Arial', 9, 'bold')).grid(
                row=0, column=col, padx=5, pady=2, sticky="w"
            )

        for i, (wt, tt, rt) in enumerate(data, 1):
            ttk.Label(table_frame, text=f"P{i}").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            ttk.Label(table_frame, text=f"{wt}").grid(row=i, column=1, padx=5, pady=2, sticky="w")
            ttk.Label(table_frame, text=f"{tt}").grid(row=i, column=2, padx=5, pady=2, sticky="w")
            ttk.Label(table_frame, text=f"{rt}").grid(row=i, column=3, padx=5, pady=2, sticky="w")

        return table_frame

    def create_summary_table(self, averages):
        summary_frame = ttk.LabelFrame(self.output_frame, text="Average Metrics", padding="10")
        summary_frame.grid(sticky="nsew", padx=5, pady=10)

        headers = ['Algorithm', 'Avg Wait Time', 'Avg Turnaround Time', 'Avg Response Time']
        for col, header in enumerate(headers):
            ttk.Label(summary_frame, text=header, font=('Arial', 9, 'bold')).grid(
                row=0, column=col, padx=10, pady=5, sticky="w"
            )

        for i, (algo, avg_wt, avg_tt, avg_rt) in enumerate(averages, 1):
            ttk.Label(summary_frame, text=algo).grid(row=i, column=0, padx=10, pady=2, sticky="w")
            ttk.Label(summary_frame, text=f"{avg_wt:.2f}").grid(row=i, column=1, padx=10, pady=2, sticky="w")
            ttk.Label(summary_frame, text=f"{avg_tt:.2f}").grid(row=i, column=2, padx=10, pady=2, sticky="w")
            ttk.Label(summary_frame, text=f"{avg_rt:.2f}").grid(row=i, column=3, padx=10, pady=2, sticky="w")

    def run_simulation(self):
        try:
            self.scheduler.processes = eval(self.process_input.get("1.0", "end").strip())
            if not all(isinstance(p, list) and len(p) == 2 for p in self.scheduler.processes):
                raise ValueError("Input must be a list of [CBT, AT] pairs.")
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input format. Error: {e}")
            return

        self.scheduler.context_switch_time = int(self.cs_input.get())
        self.scheduler.time_quantum = int(self.tq_input.get())

        for widget in self.output_frame.winfo_children():
            widget.destroy()

        algorithms = ["FCFS", "SPN", "HRRN", "RR", "SRTF", "DART"]
        averages = []

        for row_i, algo in enumerate(algorithms):
            gantt, WT, TT, RT = getattr(self.scheduler, algo.lower())()

            avg_wt = sum(WT) / len(WT)
            avg_tt = sum(TT) / len(TT)
            avg_rt = sum(RT) / len(RT)
            averages.append((algo, avg_wt, avg_tt, avg_rt))

            algo_frame = ttk.LabelFrame(self.output_frame, text=algo, padding="5")
            algo_frame.grid(row=row_i, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

            fig, ax = plt.subplots(figsize=(8, 2))

            for (start, end, process, event) in gantt:
                if event == "arrival":
                    ax.plot(start, process, "o", color="green", label="Arrival" if row_i == 0 else "")
                elif event == "execution":
                    ax.plot([start, end], [process, process], color="blue", lw=2, label="Execution" if row_i == 0 else "")
                    ax.text(start, process + 0.1, f'{start}', ha='center', va='bottom')
                    ax.text(end, process + 0.1, f'{end}', ha='center', va='bottom')
                elif event == "context_switch":
                    ax.plot([start, end], [process, process], color="red", linestyle="dotted", label="Context Switch" if row_i == 0 else "")
                    ax.text(start, process + 0.1, f'{start}', ha='center', va='bottom')
                    ax.text(end, process + 0.1, f'{end}', ha='center', va='bottom')

            ax.set_yticks(range(len(self.scheduler.processes)))
            ax.set_yticklabels([f"P{i+1}" for i in range(len(self.scheduler.processes))])
            ax.set_xlabel("Time", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, algo_frame)
            canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

            metrics_data = list(zip(WT, TT, RT))
            self.create_metrics_table(algo_frame, f"{algo} Metrics", metrics_data)

        self.create_summary_table(averages)


if __name__ == "__main__":
    root = tk.Tk()
    app = SchedulerApp(root)
    root.mainloop()
