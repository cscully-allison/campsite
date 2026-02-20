import * as d3 from "d3";
import {animate, stagger} from "animejs";
import type { AnyModel } from "@anywidget/types";
import { marked } from "marked";
import { hy } from "zod/locales";

const padding = 10;

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

    constructor(
        model: CSModel,
        svg: any,
        session_info: Object
    ) {
        this.model = model;
        this.svg = svg;
        this.session_info = session_info;
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
        const width = +this.svg.attr("width") / 2;
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


class CodeDisplayInterface{
    model:any;
    svg:any;
    dimensions:Object;
    disp_grp:any;
    text_area:any;


    constructor(model:any, svg:any, session_info:Object){
        this.model = model;
        this.svg = svg;
        this.dimensions = {};
        this.disp_grp = null;

        this.createDisplay();   
    }

    toHash(string:string) {
        return string.split('').reduce((hash, char) => {
            return char.charCodeAt(0) + (hash << 6) + (hash << 16) - hash;
        }, 0);
    }


    createDisplay(){
        const self = this;
        const width = +this.svg.attr("width")/2 - padding;
        const height = +this.svg.attr("height");

        
        const dimensions = {
            codeWindowWidth: width - 2 * padding,
            codeWindowHeight: height - 2 * padding,
            textAreaWidth: width - 4 * padding,
            textAreaHeight: height - 4 * padding
        };

        this.dimensions = dimensions;

        this.disp_grp = this.svg.append('g')
                                .attr('class', 'display-grp')
                                .attr('transform', `translate(${width+2*padding},${0})`);

        this.disp_grp.append("rect")
                .attr("x", padding)
                .attr("y", padding)
                .attr("width", dimensions.codeWindowWidth)
                .attr("height", dimensions.codeWindowHeight)
                .attr("fill", "#f9f9f9")
                .attr("stroke", "#ccc");

        this.text_area = this.disp_grp.append("foreignObject")
                            .attr("x", padding * 2)
                            .attr("y", padding * 2)
                            .attr("width", dimensions.textAreaWidth-padding)
                            .attr("height", dimensions.textAreaHeight)
                            .append("xhtml:div")
                            .style("overflow-y", "auto")
                            .style("height", `${dimensions.textAreaHeight}px`)
                            .style("width", `${dimensions.textAreaWidth-padding}px`)
                            .style("font-family", "Arial, sans-serif")
                            .style("font-size", "12px")
                            .style("color", "#333")
                            .attr("class", "chat-messages");

        this.disp_grp.append('text')
                .text('The agents will output code here for you to copy and paste. :)')
                .attr('transform', `translate(${padding*2},${padding*4})`)

    }

    render(){
        const self = this;
        this.disp_grp.selectAll('text').remove();
        // console.log("rendercalled", this.model.response);

        console.log("This model updating in render:", this.model.response);

        this.text_area.selectAll('.chat-text')
            .data([this.model.response], (d:any)=>{this.toHash(d.hypothesis)})
            .join(
                (enter:any) => {
                    let hyp_div = enter.append("div")
                         .style('text-align', "left")
                         .style('margin-bottom', "10px")
                    hyp_div.append("strong")
                            .text("Translated Hypothesis:")
                    hyp_div.append("br");
                    hyp_div.append("div")
                         .style('margin-left', "10px")
                         .style('background-color', "#f4f4f4")
                         .style('padding', "10px")
                         .style('border', "1px solid #ddd") 
                         .html((d:any)=>`${marked(d.hypothesis)}`);
                         

                    let code_div = enter.append("div")
                         .style('text-align', "left")
                         .style('margin-bottom', "10px")
                    code_div.append("strong")
                            .text("Generated Code:")
                    code_div.append("br");
                    code_div.append("div")
                         .style('margin-left', "10px")
                         .style('background-color', "#f4f4f4")
                         .style('padding', "10px")
                         .style('border', "1px solid #ddd") 
                         .html((d:any)=>`${marked(d.code)}`);
                    },
                (update:any) => {update},
                (exit:any) => {exit.remove()}
            )
    }
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
    .attr("width", 1300)
    .attr("height", 600);

  svg.style("border", "1px solid black");

  const data_model = new CSModel(data);

  const CI = new ChatInterface(data_model, svg, session_info);
  const CDI = new CodeDisplayInterface(data_model, svg, session_info);

  data_model.add_view(CI);
  data_model.add_view(CDI);

  model.on("change:_vis_data", () => {
    const updated = model.get("_vis_data");
    console.log("RE RENDER:", updated);
  });
}


export default{ render };