package termsdemo

import (
	"autosupply/bookmsg"
	"base/config"
	"base/enums"
	"base/events"
	"base/orderbook/cneq"
	"base/orderbook/ob"
	"base/security"
	"galpha/terms"
	"galpha/terms/base"
)

type TermDemo struct {
	*base.Base

	names []string // names of the fields
	outs  []float64

	isSH bool
	isSZ bool

	ob ob.IOrderBook // 自建book
}

func init() {
	cneq.UNCORSSING_ON_ADD = false // 关闭 orderbook 在 SZ 主动 add 来到时，自动uncross的功能
}

func New(cfg config.IConfig, sid int) *TermDemo {
	ret := &TermDemo{
		Base: base.New(cfg, sid),
	}

	// r.names = cfg.GetListOfStringX("names")
	// r.outs = make([]float64, len(r.names))
	ret.names = []string{"ask1", "asize1", "bid1", "bsize1", "ob.ask1", "ob.asize1", "ob.bid1", "ob.bsize1"}
	ret.outs = make([]float64, len(ret.names))

	ret.isSH = security.Lookup(sid).GetEx() == enums.ExSH
	ret.isSZ = !ret.isSH

	ret.ob = security.Lookup(sid).OrderBook()
	ret.ob.AddMsgHook(ret)

	return ret
}

func (x *TermDemo) DataType() string { return "f64" }
func (x *TermDemo) Headers() []string {
	return x.names
}

// On Msg Update ###################################################################
func (x *TermDemo) OnOBAddPre(msg *bookmsg.OBAddMsg, orderPrice float64) {}
func (x *TermDemo) OnOBCancelPre(msg *bookmsg.OBCancelMsg)               {}
func (x *TermDemo) OnOBTradePre(msg *bookmsg.OBTradeMsg)                 {}
func (x *TermDemo) OnOBReplacePre(msg *bookmsg.OBReplaceMsg)             {}
func (x *TermDemo) OnOBRebuildPre(msg *bookmsg.OBRebuildMsg)             {}

func (x *TermDemo) OnFLBEndBatchMsg(msg *bookmsg.FLBEndBatchMsg, events []events.IEvent) {
	x.outs[0] = msg.GetAsk(0)
	x.outs[1] = msg.GetAsize(0)
	x.outs[2] = msg.GetBid(0)
	x.outs[3] = msg.GetBsize(0)

	x.outs[4] = x.ob.Ask(0).Price()
	x.outs[5] = x.ob.Ask(0).Shares()
	x.outs[6] = x.ob.Bid(0).Price()
	x.outs[7] = x.ob.Bid(0).Shares()
}

func (x *TermDemo) OnOBTradeMsg(msg *bookmsg.OBTradeMsg, events []events.IEvent) {}

func (x *TermDemo) OnOBCancelMsg(msg *bookmsg.OBCancelMsg, events []events.IEvent) {}

func (x *TermDemo) OnOBAddMsg(msg *bookmsg.OBAddMsg, events []events.IEvent) {}

// On Snap ###################################################################
func (x *TermDemo) Snap(snapId int, epoch float64, out []float64) {
	copy(out, x.outs)
}

// check if TermDemo implements ITerm
var _ terms.ITerm = (*TermDemo)(nil)

// register TermDemo
func init() {
	terms.Register("term_demo", func(cfg config.IConfig, sid int) terms.ITerm { return New(cfg, sid) })
}
