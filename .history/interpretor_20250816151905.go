package main

import (
	"fmt"
	"strconv"
	"strings"
)

type NodeType int

const (
	NodePk NodeType = iota
	NodeAfter
	NodeOlder
	NodeMulti
	NodeAnd
	NodeOr
	NodeUnknown
	NodeWsh
	NodeTr
	NodeXpub
)

type Node struct {
	Type     NodeType
	Value    string  // numbers, keys, thresholds
	Children []*Node // subexpressions
}

// ---------------- Parsing ----------------

func Parse(expr string) (*Node, error) {
	expr = strings.TrimSpace(expr)

	// leaf cases
	if strings.HasPrefix(expr, "pk(") {
		return &Node{Type: NodePk, Value: inside(expr)}, nil
	}
	if strings.HasPrefix(expr, "after(") {
		return &Node{Type: NodeAfter, Value: inside(expr)}, nil
	}
	if strings.HasPrefix(expr, "older(") {
		return &Node{Type: NodeOlder, Value: inside(expr)}, nil
	}
	if strings.HasPrefix(expr, "tr(") {
		inner := inside(expr)
		// Taproot can contain multiple args (key + script tree)
		args := splitArgs(inner)
		children := []*Node{}
		for _, a := range args {
			child, err := Parse(a)
			if err != nil {
				return nil, err
			}
			children = append(children, child)
		}
		return &Node{Type: NodeTr, Children: children}, nil
	}
	if strings.HasPrefix(expr, "multi(") {
		args := splitArgs(inside(expr))
		if len(args) < 2 {
			return nil, fmt.Errorf("multi() needs at least threshold and one key")
		}
		return &Node{
			Type:  NodeMulti,
			Value: args[0],
			Children: func() []*Node {
				kids := []*Node{}
				for _, k := range args[1:] {
					kids = append(kids, &Node{Type: NodePk, Value: k})
				}
				return kids
			}(),
		}, nil
	}
	if strings.HasPrefix(expr, "wsh(") {
		inner := inside(expr)
		child, err := Parse(inner)
		if err != nil {
			return nil, err
		}
		return &Node{Type: NodeWsh, Children: []*Node{child}}, nil
	}

	// binary operators
	if strings.HasPrefix(expr, "and_v(") {
		args := splitArgs(inside(expr))
		if len(args) != 2 {
			return nil, fmt.Errorf("and_v must have 2 args")
		}
		left, _ := Parse(args[0])
		right, _ := Parse(args[1])
		return &Node{Type: NodeAnd, Children: []*Node{left, right}}, nil
	}
	if strings.HasPrefix(expr, "or_d(") {
		args := splitArgs(inside(expr))
		if len(args) != 2 {
			return nil, fmt.Errorf("or_d must have 2 args")
		}
		left, _ := Parse(args[0])
		right, _ := Parse(args[1])
		return &Node{Type: NodeOr, Children: []*Node{left, right}}, nil
	}
	if strings.HasPrefix(expr, "v:") {
		inner := expr[2:]
		return Parse(inner)
	}
	if strings.Contains(expr, "xpub") || strings.Contains(expr, "tpub") {
		return &Node{Type: NodeXpub, Value: expr}, nil
	}

	return &Node{Type: NodeUnknown, Value: expr}, nil
}

// get string inside parentheses
func inside(s string) string {
	start := strings.Index(s, "(")
	end := strings.LastIndex(s, ")")
	if start >= 0 && end > start {
		return s[start+1 : end]
	}
	return ""
}

// split by commas, respecting nested parentheses
func splitArgs(s string) []string {
	args := []string{}
	depth := 0
	cur := ""
	for _, ch := range s {
		switch ch {
		case '(':
			depth++
			cur += string(ch)
		case ')':
			depth--
			cur += string(ch)
		case ',':
			if depth == 0 {
				args = append(args, strings.TrimSpace(cur))
				cur = ""
			} else {
				cur += string(ch)
			}
		default:
			cur += string(ch)
		}
	}
	if cur != "" {
		args = append(args, strings.TrimSpace(cur))
	}
	return args
}

// ---------------- Interpreter ----------------

// convert blocks to human time
func blocksToTime(blocks int) string {
	days := float64(blocks) / 144.0
	years := days / 365.0

	if years >= 1.0 {
		return fmt.Sprintf("%.1f years (%d blocks)", years, blocks)
	}
	return fmt.Sprintf("%.0f days (%d blocks)", days, blocks)
}

func Explain(n *Node) string {
	switch n.Type {
	case NodePk:
		return fmt.Sprintf("can be spent by key %s", n.Value)

	case NodeAfter:
		blocks, err := strconv.Atoi(n.Value)
		if err == nil {
			return fmt.Sprintf("only after block %s (~%s)", n.Value, blocksToTime(blocks))
		}
		return fmt.Sprintf("only after block %s", n.Value)

	case NodeOlder:
		blocks, err := strconv.Atoi(n.Value)
		if err == nil {
			return fmt.Sprintf("only after ~%s", blocksToTime(blocks))
		}
		return fmt.Sprintf("only after %s blocks", n.Value)

	case NodeMulti:
		return fmt.Sprintf("a %s-of-%d multisig", n.Value, len(n.Children))

	case NodeAnd:
		return fmt.Sprintf("requires BOTH (%s) AND (%s)", Explain(n.Children[0]), Explain(n.Children[1]))

	case NodeOr:
		return fmt.Sprintf("can be spent by EITHER (%s) OR (%s)", Explain(n.Children[0]), Explain(n.Children[1]))

	case NodeWsh:
		return fmt.Sprintf("SegWit v0 script: %s", Explain(n.Children[0]))

	case NodeTr:
		if len(n.Children) == 1 {
			return fmt.Sprintf("Taproot key path spend:\n  %s", Explain(n.Children[0]))
		}

		parts := []string{}
		for _, c := range n.Children {
			// indent child explanations
			child := Explain(c)
			lines := strings.Split(child, "\n")
			for i, l := range lines {
				if i == 0 {
					lines[i] = "- " + l
				} else {
					lines[i] = "  " + l
				}
			}
			parts = append(parts, strings.Join(lines, "\n"))
		}

		return fmt.Sprintf("Taproot script path spend with %d options:\n%s", len(n.Children), strings.Join(parts, "\n"))

	case NodeXpub:
		desc := n.Value
		fp := ""
		if strings.HasPrefix(desc, "[") {
			end := strings.Index(desc, "]")
			if end > 0 {
				fp = desc[1:end] // fingerprint + path
			}
		}
		if fp != "" {
			return fmt.Sprintf("key at %s", fp)
		}
		return "extended public key"

	default:
		return "unrecognized miniscript"
	}
}
