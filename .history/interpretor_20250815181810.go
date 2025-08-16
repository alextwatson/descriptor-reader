package main

import (
	"fmt"
	"regexp"
	"strings"
)

// interpretDescriptor inspects a descriptor and returns a human-readable meaning
func interpretDescriptor(desc string) string {
	parts := strings.Split(desc, "#")
	core := parts[0]

	// Regex patterns for timelocks
	afterRe := regexp.MustCompile(`after\((\d+)\)`)
	olderRe := regexp.MustCompile(`older\((\d+)\)`)

	var meaning string

	switch {
	case strings.HasPrefix(core, "wpkh("):
		meaning = "Native SegWit (bech32) single key"
	case strings.HasPrefix(core, "sh(wpkh("):
		meaning = "P2SH-wrapped SegWit single key"
	case strings.HasPrefix(core, "sh("):
		meaning = "Legacy P2SH script"
	case strings.HasPrefix(core, "wsh("):
		if strings.Contains(core, "multi(") {
			meaning = "Native SegWit multisig wallet"
		} else {
			meaning = "Native SegWit custom script"
		}
	case strings.HasPrefix(core, "tr("):
		meaning = "Taproot script path spend"
	default:
		meaning = "Unknown or unsupported descriptor type"
	}

	// Check for timelocks
	if afterMatch := afterRe.FindStringSubmatch(core); afterMatch != nil {
		meaning += fmt.Sprintf(" with absolute timelock at block %s", afterMatch[1])
	}
	if olderMatch := olderRe.FindStringSubmatch(core); olderMatch != nil {
		meaning += fmt.Sprintf(" with relative timelock of %s blocks", olderMatch[1])
	}

	return meaning
}

func main() {
	tests := []string{
		"wsh(and_v(v:pk(xpub...),older(12960)))#abcd1234",
		"wsh(or_d(pk(key1),and_v(v:pk(key2),after(500000))))#deadbeef",
		"tr([f00dbabe/86h/0h/0h]xpub.../0/*)#c0ffee12",
	}

	for _, d := range tests {
		fmt.Printf("%s\n  -> %s\n\n", d, interpretDescriptor(d))
	}
}
