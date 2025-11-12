import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
from tqdm import tqdm

"""

GUI Tool to Extract Unique Rows from Excel Files
------------------------------------------------

Features:
---------
- Detects and removes duplicate rows (based on all columns)
- Lets user select specific columns to extract
- Optionally assigns unique IDs to each row
- Supports flexible header row selection and multiple sheets

Usage:
------
    python excel_extractor.py
"""

class UniqueExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel Unique Row Extractor")
        self.root.geometry("700x650")

        self.file_path = None
        self.sheet_name = None
        self.df = None
        self.column_vars = {}
        self.header_row = tk.IntVar(value=0)

        # File selection
        ttk.Button(root, text="Select Excel File", command=self.load_excel).pack(pady=10)

        # Sheet selection
        self.sheet_label = ttk.Label(root, text="Available Sheets:")
        self.sheet_label.pack()
        self.sheet_box = ttk.Combobox(root, state="readonly", width=40)
        self.sheet_box.pack(pady=5)
        self.sheet_box.bind("<<ComboboxSelected>>", self.load_columns)

        # Header row selection
        header_frame = ttk.Frame(root)
        header_frame.pack(pady=5)
        ttk.Label(header_frame, text="Header row index (0-based):").pack(side="left", padx=5)
        header_entry = ttk.Entry(header_frame, textvariable=self.header_row, width=6)
        header_entry.pack(side="left")
        ttk.Button(header_frame, text="Reload Columns", command=self.load_columns).pack(side="left", padx=5)

        # scrollable frame for column checkboxes
        self.columns_container = ttk.LabelFrame(root, text="Select Columns to Extract")
        self.columns_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.columns_container)
        self.scrollbar = ttk.Scrollbar(self.columns_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # --- Unique ID Option ---
        self.add_id_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            root,
            text="Assign Unique IDs to each row",
            variable=self.add_id_var
        ).pack(pady=5)

        self.extract_btn = ttk.Button(root, text="Extract Unique Rows", command=self.extract_unique_rows)
        self.extract_btn.pack(pady=10)

    # Core methods
    def load_excel(self):
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel Files", "*.xlsx *.xls")]
        )
        if not file_path:
            return
        self.file_path = file_path
        tqdm.write(f"Loaded file: {file_path}")

        try:
            xls = pd.ExcelFile(file_path)
            sheets = xls.sheet_names
            self.sheet_box["values"] = sheets
            if sheets:
                self.sheet_box.current(0)
                self.load_columns()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read Excel file:\n{e}")

    def load_columns(self, event=None):
        if not self.file_path:
            return

        self.sheet_name = self.sheet_box.get()
        if not self.sheet_name:
            return

        try:
            # Try to guess header row if not set
            df_preview = pd.read_excel(self.file_path, sheet_name=self.sheet_name, header=None, nrows=10)
            possible_headers = [
                i for i, row in df_preview.iterrows()
                if row.notna().sum() >= len(row) / 2
            ]
            if possible_headers and self.header_row.get() == 0:
                guessed_row = possible_headers[0]
                self.header_row.set(guessed_row)
                tqdm.write(f"Guessed header row: {guessed_row}")

            df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=self.header_row.get()
            )
            self.df = df

            # Clear old checkboxes
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()

            # Add new checkboxes
            self.column_vars.clear()
            for col in df.columns:
                var = tk.BooleanVar(value=False)
                chk = ttk.Checkbutton(self.scrollable_frame, text=col, variable=var)
                chk.pack(anchor="w", padx=10)
                self.column_vars[col] = var

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sheet:\n{e}")

    def extract_unique_rows(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please select a file and sheet first.")
            return

        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one column.")
            return

        tqdm.write(f"Processing '{self.sheet_name}' with columns: {selected_columns}")

        duplicate_rows = self.df[self.df.duplicated(keep=False)]
        if not duplicate_rows.empty:
            messagebox.showinfo(
                "Duplicates Found",
                f"Found {len(duplicate_rows)} duplicate rows.\nThey will be shown in console.",
            )
            print("\n--- Duplicates ---\n", duplicate_rows)
        else:
            tqdm.write("No duplicates found.")

        df_unique = self.df.drop_duplicates(keep="first")
        df_selected = df_unique[selected_columns].copy()

        if self.add_id_var.get():
            df_selected.insert(0, "Unique_ID", range(1, len(df_selected) + 1))
            tqdm.write("Unique IDs assigned to each row.")
        else:
            tqdm.write("Unique ID assignment skipped.")

        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            title="Save Extracted File As"
        )
        if not save_path:
            return

        try:
            df_selected.to_excel(save_path, index=False)
            messagebox.showinfo(
                "Success",
                f"Saved {len(df_selected)} unique rows to:\n{save_path}",
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")


def main():
    root = tk.Tk()
    app = UniqueExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
