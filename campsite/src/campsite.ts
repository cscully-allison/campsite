import * as d3 from "d3";
import {animate, stagger} from "animejs";
import type { AnyModel } from "@anywidget/types";
import { marked } from "marked";
import { hy } from "zod/locales";

const padding = 10;


/**
 * Insert hypothesis (markdown) and code cells into the notebook.
 * Primary path: JupyterLab shared-model API for immediate insertion.
 * Fallback: sends a message to the Python widget for deferred insertion.
 */
function insertNotebookCells(
    widgetModel: WidgetModel,
    widgetEl: HTMLElement,
    hypothesis: string,
    code: string
): void {
    try {
        // JupyterLab path: access notebook shared model via widget manager internals
        const wm = (widgetModel as any).widget_manager;
        const sharedModel = wm._context.model.sharedModel;

        // Find the index of the cell containing this widget
        const cellEl = widgetEl.closest('.jp-Cell');
        let cellIndex = -1;
        if (cellEl) {
            const notebook = cellEl.closest('.jp-Notebook');
            if (notebook) {
                const cells = notebook.querySelectorAll('.jp-Cell');
                cells.forEach((cell: Element, idx: number) => {
                    if (cell === cellEl) cellIndex = idx;
                });
            }
        }

        // Insert after the widget cell (or at the end if we couldn't find it)
        const insertIndex = cellIndex >= 0 ? cellIndex + 1 : sharedModel.cells.length;

        sharedModel.insertCells(insertIndex, [
            { cell_type: 'markdown', source: `## Translated Hypothesis\n${hypothesis}`, metadata: {} },
            { cell_type: 'code', source: code, metadata: {} },
        ]);

        console.log("Inserted cells via JupyterLab shared model");
    } catch (e) {
        // Fallback: send to Python for deferred cell creation
        console.warn("JupyterLab cell insertion failed, falling back to Python:", e);
        widgetModel.send({
            type: "create_cells",
            hypothesis: hypothesis,
            code: code,
        });
    }
}


class CSModel{
    data:any;
    response: any;
    views:any[];

    constructor(data:any){
        this.data = data;
        this.response = null;

        this.views = [];
    }

    update_response(resp:any){
        this.response =  resp;
        this.update();
    }

    add_view(view:any){
        this.views.push(view);
    }

    update(){
        for(let view of this.views){
            view.render();
        }
    }

}


class ChatInterface {
    model: CSModel;
    svg: any;
    session_info: any;
    awaiting_response: boolean;
    anywidget_model: WidgetModel;
    widget_el: HTMLElement;

    constructor(
        model: CSModel,
        svg: any,
        session_info: Object,
        anywidget_model: WidgetModel,
        widget_el: HTMLElement
    ) {
        this.model = model;
        this.svg = svg;
        this.session_info = session_info;
        this.anywidget_model = anywidget_model;
        this.widget_el = widget_el;
        this.createChatInterface();
        this.awaiting_response = false;
    }

    async manageUserTextInput(this_evnt:any): Promise<void> {
        const self = this;
        this.awaiting_response = true;
        d3.select('#send-button').attr("disabled", "true");
        const inputSel = d3.select(".chat-input");
        const messagesDiv = document.querySelector(".chat-messages") as HTMLDivElement;
        const userMessage = (inputSel.property("value") as string).trim();
        if (userMessage === "") return;

        let userMessageBlock = d3.select(messagesDiv)
            .append("div")
            .style("text-align", "left")
            .style("margin-bottom", "5px")
            .style("padding", "5px")
            .style("border", "1px solid #585858")
            .style("border-radius", "5px");
    
        userMessageBlock.append("strong")
            .text(`You:`);
        userMessageBlock.append("br");
        userMessageBlock.append("div")
            .style("margin-left", "10px")
            .style("white-space", "pre-wrap")
            .style("color", "black")
            .html(`${marked(userMessage)}`);



        inputSel.property("value", "");

        const placeholderSel = d3.select(messagesDiv)
            .append("div")
            .style("text-align", "left")
            .style("margin-bottom", "5px")
            .style("height", "40px");

        const placeholder = placeholderSel.node() as HTMLDivElement;

        const loading = d3.select("#loading");
        loading.attr("visibility", "visible");

        
        console.log("Sending user message to server:", self.model.data);
        const response = await fetch(self.session_info["endpoint"] + "/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                sessionId: self.session_info["session_id"],
                question: userMessage,
                dataSummary: self.model.data
            }),
        }).then(res => {return res.json()});

        console.log("Fetch response:", response);


        // let resp = await self.routeMessages(userMessage);
        // console.log(resp);


        if (!response["waiting"]) {
            this.model.update_response(response);
            insertNotebookCells(
                this.anywidget_model,
                this.widget_el,
                response["hypothesis"],
                response["code"]
            );
        }
        else{    
            let responseBlock = d3.select(messagesDiv).append("div")
                .style('text-align', "left")
                .style('margin-bottom', "10px")
                .style("border", "1px solid #585858")
                .style("border-radius", "5px");
            
            responseBlock.append("strong").text("LLM Response:");
            responseBlock.append("br");
            responseBlock.append("div")
                .style('margin-left', "10px")
                .style('color', "black")
                .html(`${marked(response["userPrompt"])}`);
        }

        d3.select('#send-button').attr("disabled", null);

        messagesDiv.removeChild(placeholder);
        loading.attr("visibility", "hidden");

        messagesDiv.scrollTop = messagesDiv.scrollHeight;

        this.awaiting_response = false;
    }


    createChatInterface(): void {
        
        const self = this;

        const buttonHeight = 30;
        const width = +this.svg.attr("width");
        const height = +this.svg.attr("height") - buttonHeight;

        const dimensions = {
            chatWidth: width - 2 * padding,
            chatHeight: height - 2 * padding,
            messagesHeight: height -  8 * padding,
            textAreaWidth: width - 4 * padding,
            inputHeight: padding * 2,
            buttonHeight: buttonHeight,
        };

        const chatGroup = this.svg.append("g").attr("class", "chat-interface");

        chatGroup
            .append("rect")
            .attr("x", padding)
            .attr("y", padding)
            .attr("width", dimensions.chatWidth)
            .attr("height", dimensions.chatHeight)
            .attr("fill", "#f9f9f9")
            .attr("stroke", "#ccc");

        chatGroup
            .append("foreignObject")
            .attr("x", padding * 2)
            .attr("y", padding * 2)
            .attr("width", dimensions.textAreaWidth - padding)
            .attr("height", dimensions.messagesHeight)
            .append("xhtml:div")
            .style("overflow-y", "auto")
            .style("height", `${dimensions.messagesHeight}px`)
            .style("width", `${dimensions.textAreaWidth - 2*padding}px`)
            .style("font-family", "Arial, sans-serif")
            .style("font-size", "12px")
            .style("color", "#333")
            .attr("class", "chat-messages");

        chatGroup
            .append("foreignObject")
            .attr("x", padding * 2)
            .attr("y", height - padding * 6)
            .attr("width", width - 4 * padding)
            .attr("height", padding * 4)
            .append("xhtml:textarea")
            .attr("wrap", "soft")
            .style("width", `${width - 4 * padding}px`)
            .style("height", `${padding * 4}px`)
            .style("font-family", "Arial, sans-serif")
            .style("font-size", "12px")
            .style("border", "1px solid #ccc")
            .style("padding", "2px")
            .attr("class", "chat-input")
            .attr("placeholder", "ex. What can you tell me about my data?")
            .on("keydown", function (event: KeyboardEvent) {
                if (event.key === "Enter" && !event.shiftKey && !self.awaiting_response) {
                    event.preventDefault();
                    self.manageUserTextInput(event);
                }
            });


        chatGroup
            .append("foreignObject")
            .attr("x", padding * 2)
            .attr("y", height - padding)
            .attr("width", width - 4 * padding)
            .attr("height", buttonHeight)
            .append("xhtml:button")
            .style("width", `${width - 4 * padding}px`)
            .style("height", `${buttonHeight}px`)
            .attr('id', 'send-button')
            .text("Send")
            .on("click", self.manageUserTextInput.bind(this));

        const loading = chatGroup.append("g").attr("id", "loading").attr("transform", `translate(${dimensions.chatWidth/2}, ${dimensions.messagesHeight})`);

        let dots = [];
        for (let x = 0; x < 3; x++) {
            const dot = loading
                .append("circle")
                .attr("class", "loading-dot")
                .attr("cx", 0 + x * 30)
                .attr("cy", 0)
                .attr("r", 8)
                .attr("fill", "grey");
            dots.push(dot.node());
        }

        animate(dots, {
            translateY: [
                { value: -10, duration: 500 },
                { value: 0, duration: 500 },
            ],
            easing: "easeInOutSine",
            delay: stagger(200),
            loop: true,
        });

        loading.attr("visibility", "hidden");
    }

    render(): void {}
}


// Update the WidgetModel interface to match the expected signature of the 'on' method
interface WidgetModel extends AnyModel {
  get(key: string): any;
  on(event: string, callback: (...args: any[]) => void): void;
  save_changes(): void;
}
export function render({
  model,
  el,
}: {
  model: WidgetModel;
  el: HTMLElement;
}) {
  let data = model.get("_summary_stats");
  let session_info:{session_id:string|unknown, endpoint:string|unknown} = {
    session_id: model.get("_session_id"),
    endpoint: model.get("_node_server_endpoint")
  };
  model.save_changes();

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", 650)
    .attr("height", 600);

  svg.style("border", "1px solid black");

  const data_model = new CSModel(data);

  const CI = new ChatInterface(data_model, svg, session_info, model, el);

  data_model.add_view(CI);

  model.on("change:_vis_data", () => {
    const updated = model.get("_vis_data");
    console.log("RE RENDER:", updated);
  });
}


export default{ render };