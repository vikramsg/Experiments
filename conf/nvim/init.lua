-- Options

vim.o.number = true
vim.o.relativenumber = true
vim.o.cursorline = true
vim.o.expandtab = true
vim.o.shiftwidth = 0
vim.o.tabstop = 4
vim.o.swapfile = false
vim.o.writebackup = false
vim.o.undofile = true

vim.g.mapleader = " "

-- Plugins

local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
    vim.fn.system({
        "git",
        "clone",
        "--filter=blob:none",
        "https://github.com/folke/lazy.nvim.git",
        "--branch=stable",
        lazypath,
    })
end
vim.opt.rtp:prepend(lazypath)

require("lazy").setup({
    {
        "https://github.com/junegunn/fzf.vim",
        dependencies = {
            "https://github.com/junegunn/fzf",
        },
        keys = {
            { "<Leader><Leader>", "<Cmd>Files<CR>", desc = "Find files" },
            { "<Leader>,", "<Cmd>Buffers<CR>", desc = "Find buffers" },
            { "<Leader>/", "<Cmd>Rg<CR>", desc = "Search project" },
        },
    },

    {
      "stevearc/oil.nvim",
      dependencies = { "nvim-tree/nvim-web-devicons" },  -- optional icons
      config = function()
        require("oil").setup({
          default_file_explorer = true,  -- use Oil instead of netrw on dirs
          float = false,                 -- disable floating windows entirely
          columns = { "icon" },          -- show only icons; add size/mtime if desired
          view_options = {
            show_hidden = true,          -- display dotfiles
            is_hidden_file = function(name)
              return vim.startswith(name, ".")
            end,
            sort = { { "type", "asc" }, { "name", "asc" } },
          },
          buf_options = {
            buflisted = false,
            bufhidden = "hide",
          },
          win_options = {
            wrap = false,
            signcolumn = "no",
            concealcursor = "nvic",
            conceallevel = 3,
          },
          delete_to_trash = false,
          skip_confirm_for_simple_edits = false,
          prompt_save_on_select_new_entry = true,
          use_default_keymaps = true,
        })
      end,
      keys = {
        { "-", "<Cmd>Oil<CR>", desc = "Browse files from here" },
        -- Note - usings 30vsplit to use 30 columns only
        { "<Leader>e", "<Cmd>30vsplit | Oil<CR>", desc = "Open Oil in left split" },
      },
    },

 })
